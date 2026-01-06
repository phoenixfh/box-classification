"""独立的ONNX导出模块 - 从predict.py中提取"""

import os
import sys
import torch
import time
from pathlib import Path
from fastai.vision.all import *

# 导入自定义模型模块
sys.path.insert(0, str(Path(__file__).parent))
try:
    from models import get_model, is_custom_model
    CUSTOM_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  自定义模块导入失败: {e}")
    CUSTOM_MODELS_AVAILABLE = False
    is_custom_model = lambda x: False


def detect_dataset_structure(data_dir):
    """检测数据集的组织结构，支持不同的目录结构模式
    
    支持的结构:
    1. data_dir/train/class1, data_dir/train/class2, ...
    2. data_dir/val/class1, data_dir/val/class2, ...
    3. data_dir/class1/train, data_dir/class2/train, ...
    4. data_dir/class1/val, data_dir/class2/val, ...
    
    Returns:
        tuple: (structure_type, classes)
        structure_type: 1 = 数据集按子集分组, 2 = 数据集按类别分组
        classes: 类别列表
    """
    data_dir = Path(data_dir)
    
    # 检查是否存在train/val子目录，结构类型1
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    
    if val_dir.exists() and val_dir.is_dir():
        # 检查val目录下是否有类别子目录
        class_dirs = [d for d in val_dir.iterdir() if d.is_dir()]
        if class_dirs:
            classes = [d.name for d in class_dirs]
            print(f"检测到数据集结构: data_dir/val/class, 类别: {classes}")
            return 1, classes
    
    if train_dir.exists() and train_dir.is_dir():
        # 检查train目录下是否有类别子目录
        class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
        if class_dirs:
            classes = [d.name for d in class_dirs]
            print(f"检测到数据集结构: data_dir/train/class, 类别: {classes}")
            return 1, classes
    
    # 检查是否有类别目录，每个类别目录下有train/val，结构类型2
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    for subdir in subdirs:
        train_subdir = subdir / "train"
        val_subdir = subdir / "val"
        if (train_subdir.exists() and train_subdir.is_dir()) or \
           (val_subdir.exists() and val_subdir.is_dir()):
            classes = [d.name for d in subdirs]
            print(f"检测到数据集结构: data_dir/class/train|val, 类别: {classes}")
            return 2, classes
    
    # 尝试第三种结构，检查一级子目录下的二级子目录
    all_subdirs = []
    for subdir in subdirs:
        sub_subdirs = [d for d in subdir.iterdir() if d.is_dir()]
        all_subdirs.extend(sub_subdirs)
    
    # 从所有二级子目录收集潜在的类别
    if all_subdirs:
        potential_classes = set()
        for d in all_subdirs:
            potential_class = d.parent.name
            if potential_class not in ['train', 'val', 'test']:
                potential_classes.add(potential_class)
        
        if potential_classes:
            classes = list(potential_classes)
            print(f"检测到可能的类别: {classes}")
            return 3, classes
    
    # 找不到清晰的结构
    print("无法自动检测数据集结构")
    return 0, []


def load_model(model_path, arch, device=None, data_path=None, img_size=320):
    """加载模型（支持分类和分割）
    
    Args:
        model_path: 模型路径
        arch: 模型架构 (如 'resnet18', 'unet_seg' 等)
        device: 设备类型('cuda'或'cpu')
        data_path: 数据集路径，用于从数据目录获取类别信息
        img_size: 图像尺寸（分割模型需要）
    """
    print("尝试加载模型...")
    
    # 判断是否为分割模型
    is_segmentation = arch.lower().endswith('_seg')
    
    if is_segmentation:
        print(f"🎯 检测到分割模型: {arch}")
        return load_segmentation_model(model_path, arch, device, img_size)
    else:
        print(f"🎯 检测到分类模型: {arch}")
        return load_classification_model(model_path, arch, device, data_path)


def load_segmentation_model(model_path, arch, device=None, img_size=2048):
    """加载分割模型
    
    Args:
        model_path: 模型路径
        arch: 模型架构名称
        device: 设备类型
        img_size: 图像尺寸
    """
    from PIL import Image
    
    # 创建临时数据用于初始化
    print("创建临时分割模型...")
    temp_path = Path('.') / "temp_images"
    temp_path.mkdir(exist_ok=True)
    
    # 创建临时图像
    if not list(temp_path.glob("*.jpg")):
        img = Image.new('RGB', (img_size, img_size), color='white')
        img.save(temp_path / "dummy.jpg")
        mask = Image.new('L', (img_size, img_size), color=0)
        mask.save(temp_path / "dummy_mask.png")
    
    # 使用 get_model 创建分割模型
    if CUSTOM_MODELS_AVAILABLE:
        model = get_model(arch, n_classes=1, n_channels=3)
    else:
        raise ValueError(f"分割模型 '{arch}' 需要自定义模型模块支持")
    
    # 创建简单的数据加载器用于包装
    from torch.utils.data import Dataset, DataLoader
    
    class DummySegDataset(Dataset):
        def __len__(self):
            return 1
        def __getitem__(self, idx):
            img = torch.rand(3, img_size, img_size)
            mask = torch.zeros(img_size, img_size)
            return TensorImage(img), TensorMask(mask)
    
    dummy_ds = DummySegDataset()
    dummy_dl = DataLoader(dummy_ds, batch_size=1)
    
    from fastai.vision.all import DataLoaders
    dls = DataLoaders(dummy_dl, dummy_dl)
    
    # 创建 Learner，需要指定损失函数
    from fastai.learner import Learner
    from torch.nn import BCEWithLogitsLoss
    
    loss_func = BCEWithLogitsLoss()
    learn = Learner(dls, model, loss_func=loss_func)
    
    # 加载权重
    print(f"从 {model_path} 加载权重...")
    state_dict = torch.load(model_path, map_location='cpu' if device != 'cuda' else 'cuda')
    
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    
    learn.model.load_state_dict(state_dict)
    print("分割模型权重加载成功!")
    
    # 设置为评估模式
    learn.model.eval()
    
    return learn


def load_classification_model(model_path, arch, device=None, data_path=None):
    """加载分类模型
    
    Args:
        model_path: 模型路径
        arch: 模型架构
        device: 设备类型('cuda'或'cpu')
        data_path: 数据集路径，用于从数据目录获取类别信息
    """
    print("尝试加载模型...")
    
    # 检查是否有预定义类别
    categories = None
    predefined_classes = os.environ.get('MODEL_CLASSES', None)
    if predefined_classes:
        categories = predefined_classes.split(',')
        print(f"使用预定义类别: {categories}")
    else:
        # 如果提供了数据路径，从数据路径获取类别
        if data_path:
            try:
                # 从指定的数据路径获取类别信息
                print(f"从指定的数据路径获取类别: {data_path}")
                _, categories = detect_dataset_structure(data_path)
                if categories:
                    print(f"从数据路径获取到类别: {categories}")
                else:
                    print("从数据路径未检测到类别，将尝试其他方法...")
            except Exception as dpe:
                print(f"从指定数据路径获取类别失败: {dpe}")
                categories = None
    
    if categories is None:
        print("错误：未指定类别。请使用 --classes 指定类别，或使用 --data_path 指定数据集路径。")
        exit(1)

    # 创建一个临时数据块并构建学习器
    print("创建临时模型...")
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock(categories)),
        get_items=get_image_files,
        get_y=lambda x: categories[0]  # 默认类别，稍后会被预测覆盖
    )
    
    # 尝试创建一个临时数据加载器
    path = Path('.')
    temp_path = path / "temp_images"
    temp_path.mkdir(exist_ok=True)
    
    # 确保有至少一个图像进行初始化
    if not list(temp_path.glob("*.jpg")):
        # 创建一个空白图像
        from PIL import Image
        img = Image.new('RGB', (320, 320), color='white')
        img.save(temp_path / "dummy.jpg")

    dls = dblock.dataloaders(temp_path, bs=1)

    print(dls.vocab)
    
    # 加载权重
    print(f"从 {model_path} 加载权重...")
    state_dict = torch.load(model_path, map_location='cpu' if device != 'cuda' else 'cuda')
    
    print(f"权重键名: {state_dict.keys()}")

    # 处理可能的包装格式
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    
    # 处理 DistributedDataParallel (DDP) 的 'module.' 前缀
    if any(key.startswith('module.') for key in state_dict.keys()):
        print("检测到 DDP 模型，移除 'module.' 前缀...")
        new_state_dict = {}
        for key, value in state_dict.items():
            # 移除 'module.' 前缀
            new_key = key.replace('module.', '') if key.startswith('module.') else key
            new_state_dict[new_key] = value
        state_dict = new_state_dict
        print(f"✓ 已移除 'module.' 前缀，新的键名示例: {list(state_dict.keys())[:3]}")
    
    # 统一使用vision_learner创建模型
    print(f"✓ 使用vision_learner创建模型: {arch}")
    try:
        learn = vision_learner(dls, arch=arch, pretrained=False, n_out=len(categories))
        learn.model.load_state_dict(state_dict)
        print("✓ 模型权重加载成功!")
    except Exception as e:
        print(f"⚠️  使用vision_learner加载失败: {e}")
        print(f"   尝试检测模型类型...")
        
        # 检查是否是旧的直接Learner保存的权重（向后兼容）
        # Timm模型的键名通常是: stem.0.weight, stages.0.blocks.0.xxx（无'0.'前缀）
        # vision_learner的键名是: 0.stem.0.weight（有'0.'前缀）
        is_old_timm_model = any('stem' in key or 'stages' in key or 'blocks' in key for key in state_dict.keys()) and \
                           not any(key.startswith('0.') for key in state_dict.keys())
        
        if is_old_timm_model:
            print(f"✓ 检测到旧格式Timm模型（直接Learner保存），使用兼容模式")
            try:
                import timm
                # 直接使用timm创建模型（向后兼容）
                model = timm.create_model(arch, pretrained=False, num_classes=len(categories))
                model.load_state_dict(state_dict)
                print("✓ 旧格式Timm模型权重加载成功!")
                
                # 创建一个简单的Learner包装
                from fastai.learner import Learner
                learn = Learner(dls, model)
            except ImportError:
                print("❌ 未安装timm库")
                raise
        else:
            # 其他错误，重新抛出
            raise e
    
    return learn


def detect_input_size(model, common_sizes=[192, 224, 256, 320, 384, 448, 512, 640]):
    """通过测试常见尺寸来推断模型训练时使用的输入尺寸
    
    Args:
        model: PyTorch模型
        common_sizes: 常用的图像尺寸列表
    
    Returns:
        int: 推断出的图像尺寸，如果无法推断则返回224作为默认值
    """
    print("\n尝试自动推断模型输入尺寸...")
    model.eval()
    model.cpu()
    
    # CNN模型通常可以接受任意尺寸，我们测试常见尺寸
    # 优先尝试最常用的尺寸
    for size in common_sizes:
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, size, size)
                output = model(dummy_input)
                print(f"✓ 推断成功！模型输入尺寸: {size}x{size}")
                return size
        except Exception as e:
            # 这个尺寸可能不适合，继续尝试
            continue
    
    # 如果都失败了，返回最常用的224
    print(f"⚠️  无法自动推断，使用默认值: 224x224")
    return 224


def export_to_onnx(model_path, arch=None, output_path=None, img_size=None, device=None, data_path=None, classes=None):
    """导出模型为ONNX格式并嵌入类别信息
    
    Args:
        model_path: PyTorch模型路径 (.pth文件)
        arch: 模型架构，如果为None则从checkpoint自动读取
        output_path: ONNX模型输出路径，如果为None则使用输入路径替换后缀
        img_size: 输入图像大小，如果为None则从checkpoint自动读取
        device: 设备类型 ('cuda' 或 'cpu')
        data_path: 数据集路径，用于获取类别信息
        classes: 类别列表字符串，用逗号分隔
    
    Returns:
        str: 导出的ONNX模型路径，如果失败则返回None
    """
    print(f"\n===== 导出ONNX模型 =====")
    print(f"源模型: {model_path}")
    
    # 先尝试从checkpoint读取arch和img_size
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 读取arch
        if arch is None and isinstance(checkpoint, dict) and 'arch' in checkpoint:
            arch = checkpoint['arch']
            print(f"✓ 从checkpoint读取到 arch: {arch}")
        
        # 读取img_size
        if img_size is None and isinstance(checkpoint, dict) and 'img_size' in checkpoint:
            img_size = checkpoint['img_size']
            print(f"✓ 从checkpoint读取到 img_size: {img_size}")
    except Exception as e:
        print(f"⚠️  读取checkpoint信息失败: {e}")
    
    # 如果仍然没有arch，使用默认值
    if arch is None:
        arch = 'resnet18'
        print(f"⚠️  未找到arch信息，使用默认值: {arch}")
    
    # 设置输出路径
    if output_path is None:
        output_path = Path(model_path).with_suffix('.onnx')
    
    print(f"输出路径: {output_path}")
    
    # 判断是否为分割模型
    is_segmentation = arch.lower().endswith('_seg')
    
    # 分割模型的默认图像尺寸
    if is_segmentation and img_size is None:
        img_size = 2048
        print(f"分割任务，使用默认图像尺寸: {img_size}")
    
    # 加载模型
    start_time = time.time()
    
    try:
        # 加载模型
        learn = load_model(model_path, arch, device, data_path, img_size)
        load_time = time.time() - start_time
        print(f"模型加载耗时: {load_time*1000:.2f}ms")
        
        # 自动推断或使用指定的图像尺寸
        if img_size is None:
            img_size = detect_input_size(learn.model)
        else:
            print(f"使用读取/指定的图像尺寸: {img_size}x{img_size}")
        
        # 准备导出
        export_start = time.time()
        dummy_input = torch.randn(1, 3, img_size, img_size).cpu()
        
        # 确保模型处于评估模式，并在CPU上
        learn.model.eval().cpu()
        
        # 获取类别信息（仅用于分类模型）
        categories = None
        if not is_segmentation:
            if classes:
                categories = [c.strip() for c in classes.split(',')]
            elif hasattr(learn, 'dls') and hasattr(learn.dls, 'vocab'):
                categories = learn.dls.vocab
            else:
                # 从数据路径获取类别
                _, categories = detect_dataset_structure(data_path)
        
        # 导出模型
        torch.onnx.export(
            learn.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # 嵌入元数据到ONNX模型
        try:
            import onnx
            model_onnx = onnx.load(output_path)
            
            # 1. 添加类别信息（仅用于分类模型）
            if not is_segmentation and categories:
                meta = model_onnx.metadata_props.add()
                meta.key = "classes"
                meta.value = ",".join([str(c) for c in categories])
                
                meta = model_onnx.metadata_props.add()
                meta.key = "class_indices"
                meta.value = ",".join([f"{i}:{c}" for i, c in enumerate(categories)])
                
                print(f"✓ 已嵌入类别信息: {len(categories)} 个类别")
            
            # 2. 添加任务类型（分割模型）
            if is_segmentation:
                meta = model_onnx.metadata_props.add()
                meta.key = "task_type"
                meta.value = "segmentation"
                
                meta = model_onnx.metadata_props.add()
                meta.key = "n_classes"
                meta.value = "1"
                
                print(f"✓ 已嵌入分割任务信息")
            
            # # 3. 尝试添加timm预处理参数
            # try:
            #     import timm
            #     import timm.data
                
            #     # 检查是否是timm模型
            #     if arch in timm.list_models():
            #         print(f"\n📊 检测到Timm模型，获取预处理配置...")
                    
            #         # 创建临时模型获取预处理配置
            #         temp_model = timm.create_model(arch, pretrained=False, num_classes=0)
            #         data_config = timm.data.resolve_model_data_config(temp_model)
                    
            #         # 将预处理参数写入metadata
            #         for key, value in data_config.items():
            #             meta = model_onnx.metadata_props.add()
            #             meta.key = f"preprocessing/{key}"
            #             # 转换为字符串
            #             meta.value = str(value)
                    
            #         print(f"✓ 已嵌入预处理配置:")
            #         print(f"   - input_size: {data_config.get('input_size')}")
            #         print(f"   - mean: {data_config.get('mean')}")
            #         print(f"   - std: {data_config.get('std')}")
            #         print(f"   - interpolation: {data_config.get('interpolation')}")
            #     else:
            #         print(f"ℹ️  非Timm模型，跳过预处理配置嵌入")
            # except ImportError:
            #     print(f"ℹ️  未安装timm库，跳过预处理配置嵌入")
            # except Exception as e:
            #     print(f"⚠️  获取预处理配置失败: {e}")
            
            # 保存更新后的ONNX模型
            onnx.save(model_onnx, output_path)
            
        except Exception as e:
            print(f"⚠️  嵌入元数据失败: {e}")

        export_time = time.time() - export_start
        total_time = time.time() - start_time
        
        print(f"导出ONNX耗时: {export_time*1000:.2f}ms")
        print(f"总耗时: {total_time*1000:.2f}ms")
        print(f"ONNX模型已成功导出到: {Path(output_path).absolute()}")
        
        # 验证导出的模型
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX模型校验通过")
            
            # 仅对分类模型验证类别顺序
            if not is_segmentation:
                # PyTorch模型类别
                pth_categories = learn.dls.vocab
                
                # ONNX模型类别
                onnx_model = onnx.load(output_path)
                onnx_categories = None
                for meta in onnx_model.metadata_props:
                    if meta.key == "classes":
                        onnx_categories = meta.value.split(',')
                
                # 验证是否一致
                if pth_categories == onnx_categories:
                    print("类别顺序一致!")
                else:
                    print("类别顺序不一致!")
                    print(f"PyTorch类别: {pth_categories}")
                    print(f"ONNX类别: {onnx_categories}")
            else:
                print("分割模型导出完成，无需验证类别顺序")
                
        except ImportError:
            print("未安装onnx，跳过模型验证")
        except Exception as ve:
            print(f"ONNX模型验证失败: {ve}")
        
        return output_path
        
    except Exception as e:
        print(f"导出ONNX时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='将PyTorch模型导出为ONNX格式')
    
    parser.add_argument('--model_path', type=str, required=True, help='PyTorch模型路径 (.pth文件)')
    parser.add_argument('--arch', type=str, default=None, help='模型架构，如果不指定则从checkpoint自动读取 (默认: None，自动读取)')
    parser.add_argument('--output_path', type=str, default=None, help='ONNX模型输出路径，默认与输入路径相同但后缀为.onnx')
    parser.add_argument('--img_size', type=int, default=None, help='输入图像大小，如果不指定则从checkpoint自动读取或推断 (默认: None，自动读取)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None, 
                       help='运行设备。默认自动选择，如果有GPU则使用GPU')
    parser.add_argument('--data_path', type=str, help='数据集路径，用于获取类别信息')
    parser.add_argument('--classes', type=str, help='类别列表，用逗号分隔（例如"cat,dog,horse"）')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告：CUDA不可用，将使用CPU")
        args.device = 'cpu'
    
    # 如果提供了类别列表，解析并存储在环境变量中
    if args.classes:
        classes = [c.strip() for c in args.classes.split(',')]
        os.environ['MODEL_CLASSES'] = ','.join(classes)
        print(f"使用指定的类别列表: {classes}")
    
    # 执行导出
    export_to_onnx(
        model_path=args.model_path,
        arch=args.arch,
        output_path=args.output_path,
        img_size=args.img_size,
        device=args.device,
        data_path=args.data_path,
        classes=args.classes
    )

#!/usr/bin/env python3
"""
Verification script for PyTorch Lightning on Aurora

Run this script to verify your environment is correctly configured
for running PyTorch Lightning on Intel Aurora.

Usage:
    python verify_environment.py
"""

def check_import(module_name, friendly_name=None):
    """Check if a module can be imported."""
    if friendly_name is None:
        friendly_name = module_name

    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {friendly_name}: {version}")
        return True
    except ImportError as e:
        print(f"✗ {friendly_name}: NOT FOUND")
        print(f"  Error: {e}")
        return False

def main():
    print("="*60)
    print("PyTorch Lightning on Aurora - Environment Verification")
    print("="*60)

    all_good = True

    # Check core packages
    print("\n1. Core PyTorch Packages:")
    all_good &= check_import('torch', 'PyTorch')
    all_good &= check_import('intel_extension_for_pytorch', 'IPEX')

    print("\n2. PyTorch Lightning:")
    all_good &= check_import('pytorch_lightning', 'PyTorch Lightning')

    print("\n3. Optional Packages:")
    check_import('deepspeed', 'DeepSpeed')
    check_import('torchmetrics', 'TorchMetrics')

    # Check XPU availability
    print("\n4. XPU Device Check:")
    try:
        import torch
        if torch.xpu.is_available():
            device_count = torch.xpu.device_count()
            print(f"✓ XPU available with {device_count} devices")

            # Get device properties
            for i in range(min(device_count, 2)):  # Show first 2 devices
                try:
                    props = torch.xpu.get_device_properties(i)
                    print(f"  Device {i}: {props.name}")
                except:
                    print(f"  Device {i}: (properties not available)")
        else:
            print("✗ XPU not available")
            print("  Note: This is expected on login nodes")
            print("  XPU devices are only available on compute nodes")
    except Exception as e:
        print(f"✗ Error checking XPU: {e}")
        all_good = False

    # Check Lightning XPU accelerator
    print("\n5. Custom PyTorch Lightning XPUAcceerator Support:")
    try:
        # Try to import XPU accelerator (may not exist in older versions)
        try:
            from aurora_utils.xpu_intel import XPUAccelerator
            if XPUAccelerator.is_available():
                print("✓ Lightning XPU accelerator available")
            else:
                print("✗ Lightning XPU accelerator not available")
                print("  Note: This may be expected on login nodes")
        except ImportError:
            print("⚠ Could not import XPUAccelerator")
            print("  Your PyTorch Lightning version may not support XPU directly")
            print("  This is OK - you can use the custom strategies from aurora_utils")
    except Exception as e:
        print(f"✗ Error checking Lightning XPU support: {e}")

    # Check distributed capabilities
    print("\n6. Distributed Training with xccl backend:")
    try:
        import os
        import torch.distributed as dist
        try:
            # init torch.distributed with xccl backend
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '29500'
            dist.init_process_group(backend='xccl', init_method='env://', rank=0, world_size=1)
            # now finish dist and terminate the dist
            dist.destroy_process_group()
            xccl_flag = True
        except:
            xccl_flag = False

        if xccl_flag == True:
            print("  ✓ XCCL backend available (required for Aurora)")
        else:
            print("  ✗ XCCL backend NOT available (required for Aurora)")
            all_good = False
    except Exception as e:
        print(f"✗ Error checking distributed: {e}")
        all_good = False

    # Check aurora_utils module
    print("\n7. Aurora Utilities:")
    try:
        from aurora_utils.ddp_intel import MPIDDPStrategy, MPIEnvironment
        print("✓ Aurora DDP utilities available")
    except ImportError as e:
        print("✗ Aurora DDP utilities not found")
        print(f"  Error: {e}")
        print("  Make sure you're running from the correct directory")
        all_good = False

    try:
        from aurora_utils.deepspeed_intel import XPUDeepSpeedStrategy
        print("✓ Aurora DeepSpeed utilities available")
    except ImportError:
        print("⚠ Aurora DeepSpeed utilities not found")
        print("  This is OK if you don't need DeepSpeed")

    # Summary
    print("\n" + "="*60)
    if all_good:
        print("✅ VERIFICATION SUCCESSFUL!")
        print("Your environment is ready for PyTorch Lightning on Aurora.")
        print("\nNext steps:")
        print("1. Submit a test job: qsub scripts/submit_ddp_simple.sh")
        print("2. Check the logs: tail -f logs/*.txt")
        print("3. Adapt simple_example.py for your use case")
    else:
        print("❌ VERIFICATION FAILED")
        print("Some required components are missing.")
        print("\nPlease check:")
        print("1. Have you loaded the frameworks module?")
        print("   module load frameworks")
        print("2. Have you activated your virtual environment?")
        print("   source /path/to/your/venv/bin/activate")
        print("3. Have you installed PyTorch Lightning?")
        print("   pip install pytorch_lightning")
    print("="*60)

if __name__ == '__main__':
    main()

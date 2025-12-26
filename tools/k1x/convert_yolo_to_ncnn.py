#!/usr/bin/env python3
"""
YOLO to NCNN Model Converter

This script automates the conversion of YOLO models to NCNN format.
It handles different YOLO variants (YOLOv5, YOLOv8, YOLO11, etc.) and types (detect, cls, pose, obb, seg).

Usage:
    python tools/k1x/convert_yolo_to_ncnn.py yolov8n
    python tools/k1x/convert_yolo_to_ncnn.py yolov8n-cls
    python tools/k1x/convert_yolo_to_ncnn.py yolo11n-obb
    python tools/k1x/convert_yolo_to_ncnn.py yolov5s_6.2
    python tools/k1x/convert_yolo_to_ncnn.py --weights /path/to/custom.pt --imgsz 640

Dependencies:
  - python3
  - ultralytics (pip install ultralytics)
  - pnnx (from ncnn)
  - ncnn python package (optional, used for verification)
  - yolo CLI on PATH (from ultralytics)
"""

import os
import sys
import subprocess
import shutil
import re
import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional


EXAMPLE_MODEL_MAP: Dict[str, Dict[str, str]] = {
    # NCNN files expected by examples/yolo*.cpp -> recommended Ultralytics model name
    'yolo11': {'model_name': 'yolo11n', 'output_base': 'yolo11n'},
    'yolo11_cls': {'model_name': 'yolo11n-cls', 'output_base': 'yolo11n_cls'},
    'yolo11_obb': {'model_name': 'yolo11n-obb', 'output_base': 'yolo11n_obb'},
    'yolo11_pose': {'model_name': 'yolo11n-pose', 'output_base': 'yolo11n_pose'},
    'yolo11_seg': {'model_name': 'yolo11n-seg', 'output_base': 'yolo11n_seg'},
    'yolov8': {'model_name': 'yolov8n', 'output_base': 'yolov8n'},
    'yolov8_cls': {'model_name': 'yolov8n-cls', 'output_base': 'yolov8n_cls'},
    'yolov8_obb': {'model_name': 'yolov8n-obb', 'output_base': 'yolov8n_obb'},
    'yolov8_pose': {'model_name': 'yolov8n-pose', 'output_base': 'yolov8n_pose'},
    'yolov8_seg': {'model_name': 'yolov8n-seg', 'output_base': 'yolov8n_seg'},
    'yoloworld': {'model_name': 'yolov8s-worldv2', 'output_base': 'yolov8s_worldv2'},
    'yolov5_pnnx': {'model_name': 'yolov5s', 'output_base': 'yolov5s'},
}


class YOLOConverter:
    """Converter for YOLO models to NCNN format"""
    
    # Model type configurations
    MODEL_CONFIGS = {
        'detect': {
            'input_shapes': ['[1,3,640,640]', '[1,3,320,320]'],
            'requires_modification': True,
            'ultralytics_formats': ['yolov8', 'yolo11', 'yolov9', 'yolov10'],
        },
        'cls': {
            'input_shapes': None,  # Classification doesn't need dynamic shapes
            'requires_modification': False,
            'ultralytics_formats': ['yolov8', 'yolo11'],
        },
        'pose': {
            'input_shapes': ['[1,3,640,640]', '[1,3,320,320]'],
            'requires_modification': True,
            'ultralytics_formats': ['yolov8', 'yolo11'],
        },
        'obb': {
            'input_shapes': ['[1,3,1024,1024]', '[1,3,512,512]'],
            'requires_modification': True,
            'ultralytics_formats': ['yolov8', 'yolo11'],
        },
        'seg': {
            'input_shapes': ['[1,3,640,640]', '[1,3,320,320]'],
            'requires_modification': True,
            'ultralytics_formats': ['yolov8', 'yolo11'],
        },
        'world': {
            'input_shapes': ['[1,3,640,640]', '[1,3,320,320]'],
            'requires_modification': True,
            'ultralytics_formats': ['yolov8'],
        }
    }
    
    def __init__(
        self,
        model_name: str,
        output_dir: str = './models',
        verbose: bool = True,
        output_base_override: Optional[str] = None,
        weights_path: Optional[str] = None,
        imgsz: Optional[str] = None,
        input_shapes_override: Optional[List[str]] = None,
        work_dir: Optional[str] = None,
        int8: bool = False,
        calib_dir: Optional[str] = None,
        imagelist: Optional[str] = None,
        calib_count: int = 200,
        int8_method: str = 'kl',
        calib_threads: int = 8,
        force_int8: bool = False,
        ncnn2table_path: Optional[str] = None,
        ncnn2int8_path: Optional[str] = None
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.verbose = verbose
        self.work_dir = (Path(work_dir).expanduser() if work_dir else Path('./ncnn_conversion_temp')).resolve()
        self.output_base_override = output_base_override
        self.weights_path = Path(weights_path).expanduser().resolve() if weights_path else None
        self.imgsz = imgsz
        self.enable_int8 = bool(int8)
        self.calib_dir = Path(calib_dir).expanduser().resolve() if calib_dir else None
        self.imagelist = Path(imagelist).expanduser().resolve() if imagelist else None
        self.calib_count = calib_count
        self.int8_method = int8_method.lower() if int8_method else 'kl'
        self.calib_threads = calib_threads
        self.force_int8 = force_int8
        self.ncnn2table_path = Path(ncnn2table_path).expanduser().resolve() if ncnn2table_path else None
        self.ncnn2int8_path = Path(ncnn2int8_path).expanduser().resolve() if ncnn2int8_path else None
        self.repo_root = Path(__file__).resolve().parents[2]
        
        # Parse model name
        self.base_name, self.model_type, self.model_family = self._parse_model_name()
        self.config = dict(self.MODEL_CONFIGS[self.model_type])
        self.input_shapes_override = self._normalize_input_shapes(input_shapes_override)
        if self.input_shapes_override:
            self.config['input_shapes'] = self.input_shapes_override
        self.requires_modification = self.config['requires_modification']
        self.pt_name = f"{self.base_name}.pt"
        self.pt_path = self.work_dir / self.pt_name
        
        # YOLO11 detection prefers direct export or static PNNX without script hacks
        if self.model_family == 'yolo11' and self.model_type == 'detect':
            self.requires_modification = False
    
    def _parse_model_name(self) -> Tuple[str, str, str]:
        """Parse model name to extract base name, type, and family"""
        name = self.model_name.lower().replace('_', '-')
        
        # Determine model type (detect, cls, pose, obb, seg, world)
        model_type = 'detect'
        for mtype in self.MODEL_CONFIGS.keys():
            if name.endswith(f'-{mtype}'):
                model_type = mtype
                break
        
        # Determine model family (yolov5, yolov8, yolo11, etc.)
        model_family = 'yolov8'  # default
        if 'yolo11' in name:
            model_family = 'yolo11'
        elif 'yolov10' in name:
            model_family = 'yolov10'
        elif 'yolov9' in name:
            model_family = 'yolov9'
        elif 'yolov8' in name:
            model_family = 'yolov8'
        elif 'yolov7' in name:
            model_family = 'yolov7'
        elif 'yolov5' in name:
            model_family = 'yolov5'
        
        # Extract base name (e.g., yolov8n, yolo11m-obb)
        base_name = name
        
        return base_name, model_type, model_family

    def _normalize_input_shapes(self, shapes: Optional[List[str]]) -> Optional[List[str]]:
        """Normalize input shape strings for pnnx (e.g., 1,3,640,640 -> [1,3,640,640])."""
        if not shapes:
            return None
        normalized = []
        for shape in shapes:
            cleaned = shape.strip().replace(' ', '')
            if not cleaned:
                continue
            if cleaned.startswith('[') and cleaned.endswith(']'):
                normalized.append(cleaned)
                continue
            cleaned = cleaned.replace('x', ',').replace('X', ',')
            normalized.append(f'[{cleaned}]')
        return normalized or None
    
    @property
    def output_base(self) -> str:
        """Final file stem for param/bin outputs"""
        return self.output_base_override or self.base_name.replace('-', '_')
    
    def log(self, message: str, level: str = 'INFO'):
        """Log message if verbose mode is enabled"""
        if self.verbose:
            colors = {
                'INFO': '\033[94m',
                'SUCCESS': '\033[92m',
                'WARNING': '\033[93m',
                'ERROR': '\033[91m',
                'RESET': '\033[0m'
            }
            color = colors.get(level, colors['INFO'])
            reset = colors['RESET']
            print(f"{color}[{level}]{reset} {message}")
    
    def run_command(self, cmd: List[str], description: str, check: bool = True) -> Tuple[bool, str]:
        """Execute a shell command and return success status"""
        self.log(f"{description}...", 'INFO')
        self.log(f"Command: {' '.join(cmd)}", 'INFO')
        
        try:
            result = subprocess.run(
                cmd,
                check=check,
                capture_output=True,
                text=True,
                cwd=str(self.work_dir)
            )
            if result.returncode == 0:
                self.log(f"{description} - completed successfully", 'SUCCESS')
                return True, result.stdout
            else:
                self.log(f"{description} - failed: {result.stderr}", 'ERROR')
                return False, result.stderr
        except subprocess.CalledProcessError as e:
            self.log(f"{description} - failed: {e.stderr}", 'ERROR')
            return False, e.stderr
        except FileNotFoundError:
            self.log(f"{description} - command not found: {cmd[0]}", 'ERROR')
            return False, f"Command not found: {cmd[0]}"
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed"""
        self.log("Checking dependencies...", 'INFO')
        
        dependencies = {
            'python3': 'Python 3',
            'pnnx': 'PNNX converter'
        }
        
        missing = []
        
        for cmd, name in dependencies.items():
            if shutil.which(cmd) is None:
                missing.append(name)
        
        # Check Python packages
        try:
            import ultralytics
            self.log(f"ultralytics version: {ultralytics.__version__}", 'INFO')
        except ImportError:
            missing.append('ultralytics (pip install ultralytics)')
        
        try:
            import ncnn
            self.log("ncnn Python module found", 'INFO')
        except ImportError:
            self.log("ncnn Python module not found (optional)", 'WARNING')
        
        if missing:
            self.log(f"Missing dependencies: {', '.join(missing)}", 'ERROR')
            self.log("Install with: pip3 install -U ultralytics pnnx ncnn", 'INFO')
            return False
        
        self.log("All dependencies satisfied", 'SUCCESS')
        return True
    
    def download_model(self) -> bool:
        """Download the source YOLO model if not exists"""
        if self.weights_path:
            if not self.weights_path.exists():
                self.log(f"Local weights not found: {self.weights_path}", 'ERROR')
                return False
            if self.weights_path.resolve() != self.pt_path.resolve():
                shutil.copy(self.weights_path, self.pt_path)
                self.log(f"Copied local weights to {self.pt_path}", 'INFO')
            else:
                self.log(f"Using local weights: {self.pt_path}", 'INFO')
            return True
        
        if self.pt_path.exists():
            self.log(f"Model {self.pt_name} already exists", 'INFO')
            return True
        
        self.log(f"Downloading model {self.pt_name}...", 'INFO')
        
        # Use ultralytics to download the model
        try:
            from ultralytics import YOLO
            model = YOLO(self.base_name)
            # Model will be auto-downloaded to cache, we need to copy it
            # The model is automatically downloaded when initialized
            self.log(f"Model {self.pt_name} downloaded successfully", 'SUCCESS')
            return True
        except Exception as e:
            self.log(f"Failed to download model: {e}", 'ERROR')
            return False
    
    def export_direct_ncnn(self) -> Optional[Tuple[Path, Path]]:
        """Use Ultralytics' built-in NCNN export"""
        cmd = [
            'yolo', 'export',
            f'model={self.pt_name}',
            'format=ncnn'
        ]
        if self.imgsz:
            cmd.append(f'imgsz={self.imgsz}')
        
        success, output = self.run_command(
            cmd,
            f"Exporting {self.base_name} to NCNN (direct Ultralytics export)"
        )
        
        if not success:
            return None
        
        model_dir = self.work_dir / f"{self.base_name}_ncnn_model"
        candidates = [model_dir] if model_dir.exists() else list(self.work_dir.glob(f"{self.base_name}*_ncnn_model"))
        if not candidates:
            self.log(f"Direct export folder not found for {self.base_name}", 'ERROR')
            return None
        
        model_dir = candidates[0]
        params = list(model_dir.glob("*.ncnn.param"))
        bins = list(model_dir.glob("*.ncnn.bin"))
        
        if not params or not bins:
            self.log(f"Direct export did not produce NCNN files in {model_dir}", 'ERROR')
            return None
        
        for param_file in params:
            expected_bin = param_file.with_suffix('.bin')
            if expected_bin.exists():
                self.log(f"Found direct export outputs in {model_dir}", 'INFO')
                return param_file, expected_bin
        
        self.log("Could not pair .param and .bin from direct export", 'ERROR')
        return None
    
    def export_to_torchscript(self) -> bool:
        """Export YOLO model to TorchScript format"""
        cmd = [
            'yolo', 'export',
            f'model={self.pt_name}',
            'format=torchscript'
        ]
        if self.imgsz:
            cmd.append(f'imgsz={self.imgsz}')
        
        success, output = self.run_command(
            cmd,
            f"Exporting {self.base_name} to TorchScript"
        )
        
        if success:
            # Check if the torchscript file was created
            expected_file = self.work_dir / f"{self.base_name}.torchscript"
            if expected_file.exists():
                return True
            else:
                self.log(f"TorchScript file not found: {expected_file}", 'ERROR')
                return False
        
        return False
    
    def convert_with_pnnx_static(self, input_shapes: Optional[List[str]] = None) -> bool:
        """Convert TorchScript to NCNN with static or fixed shape"""
        torchscript_file = f"{self.base_name}.torchscript"
        
        cmd = ['pnnx', torchscript_file]
        
        if input_shapes:
            for i, shape in enumerate(input_shapes):
                key = 'inputshape' if i == 0 else f'inputshape{i+1}'
                cmd.append(f'{key}={shape}')
        
        success, output = self.run_command(
            cmd,
            "Converting to NCNN with static shape (first pass)"
        )
        
        if success:
            # Check for generated Python file
            pnnx_py = self.work_dir / f"{self.base_name.replace('-', '_')}_pnnx.py"
            base_clean = self.base_name.replace('-', '_')
            static_param = self.work_dir / f"{base_clean}.ncnn.param"
            static_bin = self.work_dir / f"{base_clean}.ncnn.bin"
            if pnnx_py.exists() or (static_param.exists() and static_bin.exists()):
                return True
            else:
                self.log(f"PNNX Python file not found: {pnnx_py}", 'ERROR')
                return False
        
        return False
    
    def modify_pnnx_script(self) -> bool:
        """Modify the PNNX-generated Python script for dynamic shape inference"""
        if not self.requires_modification:
            self.log("No modification required for this model type", 'INFO')
            return True
        
        pnnx_py_name = f"{self.base_name.replace('-', '_')}_pnnx.py"
        pnnx_py = self.work_dir / pnnx_py_name
        
        if not pnnx_py.exists():
            self.log(f"PNNX script not found: {pnnx_py}", 'ERROR')
            return False
        
        self.log(f"Modifying {pnnx_py_name} for dynamic shape inference", 'INFO')
        
        try:
            with open(pnnx_py, 'r') as f:
                content = f.read()
            
            original_content = content
            
            # Apply modifications based on model type
            if self.model_type in ['detect', 'pose', 'obb', 'seg', 'world']:
                # Pattern 1: Modify reshape for dynamic image sizes
                # v_XXX = v_YYY.view(1, N, FIXED_NUMBER)
                # -> v_XXX = v_YYY.view(1, N, -1).transpose(1, 2)
                
                pattern1 = r'(v_\d+)\s*=\s*(v_\d+)\.view\(1,\s*(\d+),\s*(\d+)\)'
                
                def replace1(match):
                    var = match.group(1)
                    source = match.group(2)
                    dim1 = match.group(3)
                    dim2 = match.group(4)
                    # Check if this is part of a concat operation (needs modification)
                    return f'{var} = {source}.view(1, {dim1}, -1).transpose(1, 2)'
                
                content = re.sub(pattern1, replace1, content)
                
                # Pattern 2: Modify concat axis
                # torch.cat((v_A, v_B, v_C), dim=2)
                # -> torch.cat((v_A, v_B, v_C), dim=1)
                content = re.sub(
                    r'torch\.cat\((.*?),\s*dim=2\)',
                    r'torch.cat(\1, dim=1)',
                    content
                )
                
                # Pattern 3: Add return statement for outputs (remove post-processing)
                # Find the last concat operations and add return
                # This is model-specific, so we'll add a generic return at the end
                
                # For YOLO11 with attention mechanism, modify area attention
                if self.model_family == 'yolo11' and self.model_type == 'detect':
                    # Modify attention view operations
                    content = re.sub(
                        r'\.view\(1,\s*128,\s*20,\s*20\)',
                        '.view(1, 128, v_95.size(2), v_95.size(3))',
                        content
                    )
                    content = re.sub(
                        r'\.reshape\(1,\s*128,\s*20,\s*20\)',
                        '.reshape(1, 128, v_95.size(2), v_95.size(3))',
                        content
                    )
            
            # Check if modifications were made
            if content != original_content:
                # Backup original file
                backup_path = pnnx_py.with_suffix('.py.bak')
                shutil.copy(pnnx_py, backup_path)
                self.log(f"Backup created: {backup_path}", 'INFO')
                
                # Write modified content
                with open(pnnx_py, 'w') as f:
                    f.write(content)
                
                self.log("PNNX script modified successfully", 'SUCCESS')
                self.log("NOTE: Automatic modifications applied. You may need to manually adjust the script.", 'WARNING')
            else:
                self.log("No modifications made (pattern not found or already modified)", 'WARNING')
            
            return True
            
        except Exception as e:
            self.log(f"Failed to modify PNNX script: {e}", 'ERROR')
            return False
    
    def re_export_torchscript(self) -> bool:
        """Re-export the modified Python model to TorchScript"""
        pnnx_module_name = f"{self.base_name.replace('-', '_')}_pnnx"
        
        cmd = [
            'python3', '-c',
            f'import {pnnx_module_name}; {pnnx_module_name}.export_torchscript()'
        ]
        
        success, output = self.run_command(
            cmd,
            "Re-exporting modified model to TorchScript"
        )
        
        if success:
            expected_file = self.work_dir / f"{pnnx_module_name}.py.pt"
            if expected_file.exists():
                return True
            else:
                self.log(f"Re-exported TorchScript not found: {expected_file}", 'WARNING')
                # For classification models, re-export might not be needed
                if self.model_type == 'cls':
                    return True
                return False
        
        return False
    
    def convert_with_pnnx_dynamic(self) -> bool:
        """Convert modified TorchScript to NCNN with dynamic shape (second pass)"""
        config = self.config
        
        if not self.requires_modification:
            # For models that don't need modification, use the original torchscript
            torchscript_file = f"{self.base_name}.torchscript"
            cmd = ['pnnx', torchscript_file]
        else:
            # Use the re-exported torchscript
            pnnx_module_name = f"{self.base_name.replace('-', '_')}_pnnx"
            torchscript_file = f"{pnnx_module_name}.py.pt"
            
            cmd = ['pnnx', torchscript_file]
            
            # Add input shapes if specified
            if config['input_shapes']:
                for i, shape in enumerate(config['input_shapes']):
                    if i == 0:
                        cmd.append(f'inputshape={shape}')
                    else:
                        cmd.append(f'inputshape{i+1}={shape}')
        
        success, output = self.run_command(
            cmd,
            "Converting to NCNN with dynamic shape (second pass)"
        )
        
        if success:
            # Check for generated NCNN files
            if self.requires_modification:
                pnnx_module_name = f"{self.base_name.replace('-', '_')}_pnnx"
                param_file = self.work_dir / f"{pnnx_module_name}.py.ncnn.param"
                bin_file = self.work_dir / f"{pnnx_module_name}.py.ncnn.bin"
            else:
                base_clean = self.base_name.replace('-', '_')
                param_file = self.work_dir / f"{base_clean}.ncnn.param"
                bin_file = self.work_dir / f"{base_clean}.ncnn.bin"
            
            if param_file.exists() and bin_file.exists():
                return True
            else:
                self.log(f"NCNN files not found: {param_file}, {bin_file}", 'ERROR')
                return False
        
        return False
    
    def rename_output_files(self, src_param: Optional[Path] = None, src_bin: Optional[Path] = None) -> bool:
        """Rename or copy output files to final names"""
        # Determine source file names
        if src_param is None or src_bin is None:
            if self.requires_modification:
                pnnx_module_name = f"{self.base_name.replace('-', '_')}_pnnx"
                src_param = self.work_dir / f"{pnnx_module_name}.py.ncnn.param"
                src_bin = self.work_dir / f"{pnnx_module_name}.py.ncnn.bin"
            else:
                base_clean = self.base_name.replace('-', '_')
                src_param = self.work_dir / f"{base_clean}.ncnn.param"
                src_bin = self.work_dir / f"{base_clean}.ncnn.bin"
        
        if not src_param.exists():
            self.log(f"NCNN param file not found: {src_param}", 'ERROR')
            return False
        if not src_bin.exists():
            self.log(f"NCNN bin file not found: {src_bin}", 'ERROR')
            return False
        
        # Determine destination file names
        output_base = self.output_base
        dst_param = self.output_dir / f"{output_base}.ncnn.param"
        dst_bin = self.output_dir / f"{output_base}.ncnn.bin"
        
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            shutil.copy(src_param, dst_param)
            shutil.copy(src_bin, dst_bin)
            
            self.log(f"Output files created:", 'SUCCESS')
            self.log(f"  {dst_param}", 'SUCCESS')
            self.log(f"  {dst_bin}", 'SUCCESS')
            
            return True
        except Exception as e:
            self.log(f"Failed to copy output files: {e}", 'ERROR')
            return False
    
    def verify_conversion(self) -> bool:
        """Verify that the conversion was successful"""
        self.log("Verifying conversion...", 'INFO')
        
        output_base = self.output_base
        param_file = self.output_dir / f"{output_base}.ncnn.param"
        bin_file = self.output_dir / f"{output_base}.ncnn.bin"
        
        if not param_file.exists():
            self.log(f"Param file not found: {param_file}", 'ERROR')
            return False
        
        if not bin_file.exists():
            self.log(f"Bin file not found: {bin_file}", 'ERROR')
            return False
        
        # Check param file format
        try:
            with open(param_file, 'r') as f:
                first_line = f.readline().strip()
                if not first_line.startswith('7767517'):
                    self.log(f"Invalid param file format (magic number)", 'ERROR')
                    return False
                
                # Count layers
                second_line = f.readline().strip()
                parts = second_line.split()
                if len(parts) >= 2:
                    num_layers = int(parts[0])
                    num_blobs = int(parts[1])
                    self.log(f"Model has {num_layers} layers and {num_blobs} blobs", 'INFO')
        except Exception as e:
            self.log(f"Failed to read param file: {e}", 'ERROR')
            return False
        
        # Check bin file size
        bin_size = bin_file.stat().st_size
        if bin_size < 1000:  # Models should be at least 1KB
            self.log(f"Bin file seems too small: {bin_size} bytes", 'WARNING')
        else:
            self.log(f"Bin file size: {bin_size / 1024 / 1024:.2f} MB", 'INFO')
        
        # Try to load with ncnn (if available)
        try:
            import ncnn
            net = ncnn.Net()
            ret = net.load_param(str(param_file))
            if ret != 0:
                self.log(f"Failed to load param file with ncnn (error code: {ret})", 'ERROR')
                return False
            
            ret = net.load_model(str(bin_file))
            if ret != 0:
                self.log(f"Failed to load model file with ncnn (error code: {ret})", 'ERROR')
                return False
            
            self.log("Model loaded successfully with ncnn", 'SUCCESS')
        except ImportError:
            self.log("ncnn Python module not available, skipping load test", 'WARNING')
        except Exception as e:
            self.log(f"Failed to load model with ncnn: {e}", 'ERROR')
            return False
        
        self.log("Conversion verified successfully", 'SUCCESS')
        return True

    def _infer_imgsz(self) -> int:
        if not self.imgsz:
            return 640
        text = str(self.imgsz).strip()
        if not text:
            return 640
        for sep in (',', 'x', 'X'):
            if sep in text:
                text = text.split(sep)[0]
                break
        try:
            value = int(float(text))
            return value if value > 0 else 640
        except ValueError:
            return 640

    def _resolve_tool(self, name: str, override: Optional[Path], extra_candidates: List[Path]) -> Optional[str]:
        if override:
            if override.exists():
                return str(override)
            self.log(f"Requested tool not found: {override}", 'ERROR')
            return None
        which = shutil.which(name)
        if which:
            return which
        for candidate in extra_candidates:
            if candidate.exists():
                return str(candidate)
        self.log(f"Required tool not found: {name}", 'ERROR')
        return None

    def _prepare_imagelist(self) -> Optional[Path]:
        if self.imagelist:
            if not self.imagelist.exists():
                self.log(f"Imagelist not found: {self.imagelist}", 'ERROR')
                return None
            return self.imagelist.resolve()

        if not self.calib_dir:
            self.log("INT8 requires --calib-dir or --imagelist", 'ERROR')
            return None
        if not self.calib_dir.exists():
            self.log(f"Calibration dir not found: {self.calib_dir}", 'ERROR')
            return None
        if self.calib_count <= 0:
            self.log("calib-count must be > 0", 'ERROR')
            return None

        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [p.resolve() for p in self.calib_dir.rglob('*') if p.suffix.lower() in exts]
        images = sorted(images, key=lambda p: str(p))
        if not images:
            self.log(f"No calibration images found in {self.calib_dir}", 'ERROR')
            return None
        images = images[: self.calib_count]

        imagelist = (self.work_dir / "imagelist.txt").resolve()
        with open(imagelist, 'w') as f:
            for path in images:
                f.write(f"{path}\n")

        self.log(f"Calibration images: {len(images)}", 'INFO')
        self.log(f"Imagelist saved to: {imagelist}", 'INFO')
        return imagelist

    def _sanitize_int8_table(self, table_file: Path) -> bool:
        try:
            lines = table_file.read_text().splitlines()
        except Exception as e:
            self.log(f"Failed to read table file: {e}", 'ERROR')
            return False

        changed = False
        replaced = 0
        sanitized_lines = []
        for line in lines:
            tokens = line.strip().split()
            if len(tokens) <= 1:
                sanitized_lines.append(line)
                continue

            max_finite = None
            for token in tokens[1:]:
                try:
                    value = float(token)
                except ValueError:
                    continue
                if math.isfinite(value):
                    max_finite = value if max_finite is None else max(max_finite, abs(value))

            if max_finite is None:
                max_finite = 1.0

            new_tokens = [tokens[0]]
            for token in tokens[1:]:
                try:
                    value = float(token)
                except ValueError:
                    value = float('nan')

                if math.isfinite(value):
                    new_tokens.append(token)
                else:
                    new_tokens.append(f"{max_finite:.6f}")
                    changed = True
                    replaced += 1

            sanitized_lines.append(" ".join(new_tokens))

        if changed:
            table_file.write_text("\n".join(sanitized_lines) + "\n")
            self.log(f"Sanitized {replaced} non-finite scale values in {table_file}", 'WARNING')

        return True

    def quantize_int8(self) -> bool:
        if not self.enable_int8:
            return True

        self.log("Starting INT8 quantization...", 'INFO')

        if self.calib_threads <= 0:
            self.log("calib-threads must be > 0", 'ERROR')
            return False
        if self.int8_method not in ('kl', 'aciq'):
            self.log(f"Unsupported int8 method: {self.int8_method}", 'ERROR')
            return False

        imagelist = self._prepare_imagelist()
        if imagelist is None:
            return False
        imagelist = imagelist.resolve()

        ncnn2table = self._resolve_tool(
            'ncnn2table',
            self.ncnn2table_path,
            [self.repo_root / 'build-host-tools/tools/quantize/ncnn2table']
        )
        ncnn2int8 = self._resolve_tool(
            'ncnn2int8',
            self.ncnn2int8_path,
            [self.repo_root / 'build-host-tools/tools/quantize/ncnn2int8']
        )
        if not ncnn2table or not ncnn2int8:
            return False

        output_base = self.output_base
        param_file = (self.output_dir / f"{output_base}.ncnn.param").resolve()
        bin_file = (self.output_dir / f"{output_base}.ncnn.bin").resolve()
        if not param_file.exists() or not bin_file.exists():
            self.log(f"Base model files not found: {param_file} {bin_file}", 'ERROR')
            return False

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_param = (self.output_dir / f"{output_base}-int8.ncnn.param").resolve()
        out_bin = (self.output_dir / f"{output_base}-int8.ncnn.bin").resolve()
        if (out_param.exists() or out_bin.exists()) and not self.force_int8:
            self.log(f"INT8 outputs already exist: {out_param} {out_bin}", 'ERROR')
            self.log("Use --force to overwrite existing INT8 outputs", 'ERROR')
            return False
        if self.force_int8:
            for path in (out_param, out_bin):
                if path.exists():
                    path.unlink()

        table_file = (self.work_dir / f"{output_base}-int8.table").resolve()
        if table_file.exists():
            table_file.unlink()

        norm_val = 1.0 / 255.0
        norm_str = f"{norm_val:.10f}"
        shape_size = self._infer_imgsz()

        cmd = [
            ncnn2table,
            str(param_file),
            str(bin_file),
            str(imagelist),
            str(table_file),
            "mean=[0,0,0]",
            f"norm=[{norm_str},{norm_str},{norm_str}]",
            f"shape=[{shape_size},{shape_size},3]",
            "pixel=RGB",
            f"thread={self.calib_threads}",
            f"method={self.int8_method}",
        ]
        success, _ = self.run_command(cmd, "Generating calibration table (ncnn2table)")
        if not success:
            return False
        if not self._sanitize_int8_table(table_file):
            return False

        cmd = [
            ncnn2int8,
            str(param_file),
            str(bin_file),
            str(out_param),
            str(out_bin),
            str(table_file),
        ]
        success, _ = self.run_command(cmd, "Quantizing model to INT8 (ncnn2int8)", check=False)
        if not success and not (out_param.exists() and out_bin.exists()):
            return False
        if not success:
            self.log("ncnn2int8 returned non-zero but outputs exist; continuing", 'WARNING')

        if not out_param.exists() or not out_bin.exists():
            self.log("INT8 outputs not found after quantization", 'ERROR')
            return False

        base_size = bin_file.stat().st_size
        int8_size = out_bin.stat().st_size
        self.log(f"Base bin size: {base_size} bytes", 'INFO')
        self.log(f"INT8 bin size: {int8_size} bytes", 'INFO')
        if int8_size >= base_size:
            self.log("INT8 bin is not smaller than base bin", 'WARNING')

        self.log("INT8 outputs created:", 'SUCCESS')
        self.log(f"  {out_param}", 'SUCCESS')
        self.log(f"  {out_bin}", 'SUCCESS')
        return True
    
    def cleanup(self, keep_intermediate: bool = False):
        """Clean up temporary files"""
        if keep_intermediate:
            self.log(f"Keeping intermediate files in {self.work_dir}", 'INFO')
            return
        
        if self.work_dir.exists():
            try:
                shutil.rmtree(self.work_dir)
                self.log("Temporary files cleaned up", 'INFO')
            except Exception as e:
                self.log(f"Failed to clean up temporary files: {e}", 'WARNING')
    
    def convert(self, keep_intermediate: bool = False) -> bool:
        """Run the complete conversion pipeline"""
        self.log(f"Starting conversion of {self.model_name}", 'INFO')
        self.log(f"Model type: {self.model_type}", 'INFO')
        self.log(f"Model family: {self.model_family}", 'INFO')
        
        # Create work directory
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Check dependencies
            if not self.check_dependencies():
                return False
            
            # Step 2: Download model
            if not self.download_model():
                return False

            # Step 3: Try direct NCNN export for supported combos
            direct_supported = self.model_family == 'yolo11' and self.model_type == 'detect'
            if direct_supported and self.enable_int8:
                self.log("INT8 enabled; skipping direct export and using PNNX pipeline", 'INFO')
            if direct_supported and not self.enable_int8:
                direct_outputs = self.export_direct_ncnn()
                if direct_outputs:
                    if not self.rename_output_files(src_param=direct_outputs[0], src_bin=direct_outputs[1]):
                        return False
                    
                    if not self.verify_conversion():
                        self.log("Verification failed after direct export, but files were created", 'WARNING')

                    if self.enable_int8:
                        if not self.quantize_int8():
                            return False
                    
                    self.log(f"Conversion of {self.model_name} completed successfully!", 'SUCCESS')
                    return True
                else:
                    self.log("Direct NCNN export failed, falling back to PNNX pipeline", 'WARNING')
            
            # Step 4: Export to TorchScript
            if not self.export_to_torchscript():
                return False
            
            use_static_only = self.model_family == 'yolo11' and self.model_type == 'detect'
            static_shapes = self.input_shapes_override
            
            # Step 5: Convert with PNNX (static shape)
            if use_static_only:
                if not static_shapes:
                    static_shapes = ['[1,3,640,640]']
                if not self.convert_with_pnnx_static(input_shapes=static_shapes):
                    return False
            else:
                if not self.convert_with_pnnx_static(input_shapes=static_shapes):
                    return False
                
                if self.requires_modification:
                    # Step 6: Modify PNNX script
                    if not self.modify_pnnx_script():
                        self.log("Script modification failed, continuing anyway...", 'WARNING')
                    
                    # Step 7: Re-export to TorchScript
                    if not self.re_export_torchscript():
                        self.log("Re-export failed, trying to proceed anyway...", 'WARNING')
                
                # Step 8: Convert with PNNX (dynamic shape)
                if not self.convert_with_pnnx_dynamic():
                    return False
            
            # Step 9: Rename output files
            if not self.rename_output_files():
                return False
            
            # Step 10: Verify conversion
            if not self.verify_conversion():
                self.log("Verification failed, but files were created", 'WARNING')

            if self.enable_int8:
                if not self.quantize_int8():
                    return False
            
            self.log(f"Conversion of {self.model_name} completed successfully!", 'SUCCESS')
            return True
            
        except Exception as e:
            self.log(f"Conversion failed with exception: {e}", 'ERROR')
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Cleanup
            self.cleanup(keep_intermediate)


def main():
    parser = argparse.ArgumentParser(
        description='Convert YOLO models to NCNN format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s yolov8n                 # Convert YOLOv8n detection model
  %(prog)s yolov8n-cls             # Convert YOLOv8n classification model
  %(prog)s yolo11m-obb             # Convert YOLO11m OBB model
  %(prog)s yolov8n-pose            # Convert YOLOv8n pose model
  %(prog)s yolov5s                 # Convert YOLOv5s model
  %(prog)s yolov8s-world           # Convert YOLOv8 World model
  
  %(prog)s yolov8n -o ./my_models  # Specify output directory
  %(prog)s yolov8n -k              # Keep intermediate files
  %(prog)s yolov8n -q              # Quiet mode
  %(prog)s --weights ./custom.pt   # Use local .pt file
  %(prog)s yolov8n --imgsz 640     # Override export image size
        '''
    )
    
    parser.add_argument(
        'model_name',
        nargs='?',
        help='Name of the YOLO model to convert (e.g., yolov8n, yolo11m-cls, yolov8n-obb)'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        default='./models',
        help='Output directory for converted models (default: ./models)'
    )
    
    parser.add_argument(
        '-k', '--keep-intermediate',
        action='store_true',
        help='Keep intermediate files after conversion'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Quiet mode (less verbose output)'
    )

    parser.add_argument(
        '--weights',
        help='Path to a local .pt weights file (optional; copied into the work dir)'
    )

    parser.add_argument(
        '--imgsz',
        help='Image size for Ultralytics export (e.g., 640 or 640,640)'
    )

    parser.add_argument(
        '--input-shape',
        action='append',
        dest='input_shapes',
        help='Override PNNX input shape(s), e.g. 1,3,640,640 (repeatable)'
    )

    parser.add_argument(
        '--work-dir',
        default='./ncnn_conversion_temp',
        help='Temporary working directory (default: ./ncnn_conversion_temp)'
    )

    parser.add_argument(
        '--int8',
        type=int,
        choices=[0, 1],
        default=0,
        help='Enable INT8 quantization (0/1, default: 0)'
    )

    calib_group = parser.add_mutually_exclusive_group()
    calib_group.add_argument(
        '--calib-dir',
        help='Directory of calibration images (optional; used for --int8)'
    )
    calib_group.add_argument(
        '--imagelist',
        help='Path to calibration imagelist file (optional; used for --int8)'
    )

    parser.add_argument(
        '--calib-count',
        type=int,
        default=200,
        help='Number of calibration images to use when --calib-dir is set (default: 200)'
    )

    parser.add_argument(
        '--int8-method',
        choices=['kl', 'aciq'],
        default='kl',
        help='INT8 calibration method (kl or aciq, default: kl)'
    )

    parser.add_argument(
        '--calib-threads',
        type=int,
        default=8,
        help='Threads for ncnn2table calibration (default: 8)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Allow overwriting existing INT8 outputs'
    )
    
    parser.add_argument(
        '--for-example',
        dest='for_example',
        help='Match model/output names expected by examples/yolo*.cpp (e.g., yolo11, yolo11_cls, yolov8)'
    )
    
    args = parser.parse_args()

    if args.int8:
        if not args.calib_dir and not args.imagelist:
            parser.error("--int8 requires --calib-dir or --imagelist")
        if args.calib_count <= 0:
            parser.error("--calib-count must be > 0")
        if args.calib_threads <= 0:
            parser.error("--calib-threads must be > 0")
    
    example_base = args.for_example[:-4] if args.for_example and args.for_example.endswith('.cpp') else args.for_example
    mapped_model_name = None
    output_base_override = None
    
    if example_base:
        if example_base in EXAMPLE_MODEL_MAP:
            mapped = EXAMPLE_MODEL_MAP[example_base]
            mapped_model_name = mapped['model_name']
            output_base_override = mapped.get('output_base')
        else:
            parser.error(f"Unknown example name '{example_base}'. Known examples: {', '.join(sorted(EXAMPLE_MODEL_MAP.keys()))}")
    
    model_name = mapped_model_name or args.model_name

    if not model_name and args.weights:
        model_name = Path(args.weights).stem
    
    if not model_name:
        parser.error("model_name is required unless --for-example or --weights is used")
    
    # Create converter
    converter = YOLOConverter(
        model_name=model_name,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        output_base_override=output_base_override,
        weights_path=args.weights,
        imgsz=args.imgsz,
        input_shapes_override=args.input_shapes,
        work_dir=args.work_dir,
        int8=bool(args.int8),
        calib_dir=args.calib_dir,
        imagelist=args.imagelist,
        calib_count=args.calib_count,
        int8_method=args.int8_method,
        calib_threads=args.calib_threads,
        force_int8=args.force
    )
    
    # Run conversion
    success = converter.convert(keep_intermediate=args.keep_intermediate)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

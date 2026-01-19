from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime
import tempfile
import io

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = Path('/tmp/satellite_uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)

SECRET_TOKEN = os.getenv('API_SECRET_TOKEN', 'my-secret-token-12345')

def verify_token():
    auth_header = request.headers.get('Authorization', '')
    token = auth_header.replace('Bearer ', '')
    return token == SECRET_TOKEN

def cleanup_old_files():
    try:
        for file in UPLOAD_FOLDER.glob('*'):
            if file.is_file():
                age = datetime.now().timestamp() - file.stat().st_mtime
                if age > 1800:
                    file.unlink()
    except Exception as e:
        print(f"Cleanup error: {e}")

def analyze_tiff_basic(file_path):
    """Basic TIFF analysis using PIL"""
    try:
        with Image.open(file_path) as img:
            # Get basic image info
            width, height = img.size
            mode = img.mode
            
            # Count bands based on mode
            band_count_map = {
                'L': 1,      # Grayscale
                'RGB': 3,    # RGB
                'RGBA': 4,   # RGB + Alpha
                'CMYK': 4,   # CMYK
                'I': 1,      # 32-bit integer
                'F': 1       # 32-bit float
            }
            
            band_count = band_count_map.get(mode, 1)
            
            # Try to get more info
            bands_info = []
            if hasattr(img, 'getbands'):
                band_names = img.getbands()
                for i, name in enumerate(band_names, 1):
                    bands_info.append({
                        'band_number': i,
                        'name': str(name),
                        'data_type': mode,
                        'min_value': 0,
                        'max_value': 255 if mode in ['L', 'RGB', 'RGBA'] else 65535,
                        'mean_value': None,
                        'nodata': None,
                        'resolution': [1.0, 1.0]
                    })
            
            return {
                'band_count': band_count,
                'bands': bands_info,
                'width': width,
                'height': height,
                'crs': 'Unknown (basic mode)',
                'bounds': [0, 0, width, height],
                'file_size': os.path.getsize(file_path)
            }
    except Exception as e:
        raise RuntimeError(f"Error analyzing TIFF: {str(e)}")

def calculate_simple_index(file_path, index_name):
    """Simplified index calculation using PIL"""
    try:
        with Image.open(file_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB' and img.mode != 'RGBA':
                img = img.convert('RGB')
            
            # Get numpy array
            img_array = np.array(img)
            
            # Assume standard band order: R, G, B
            if len(img_array.shape) == 3:
                red = img_array[:, :, 0].astype(float)
                green = img_array[:, :, 1].astype(float)
                blue = img_array[:, :, 2].astype(float)
                
                # Simple calculations (approximation)
                if index_name == 'NDVI':
                    # Approximation: using red/green as proxy
                    result = (green - red) / (green + red + 1e-10)
                elif index_name == 'GNDVI':
                    result = (green - red) / (green + red + 1e-10)
                elif index_name == 'NDWI':
                    result = (green - blue) / (green + blue + 1e-10)
                else:
                    # Default calculation
                    result = (green - red) / (green + red + 1e-10)
                
                # Normalize to -1 to 1
                result = np.clip(result, -1, 1)
                
                return result
            else:
                raise ValueError("Image must have at least 3 bands")
    except Exception as e:
        raise RuntimeError(f"Error calculating index: {str(e)}")

def create_points_excel(data_array, output_path):
    """Convert array to point coordinates with values"""
    height, width = data_array.shape
    
    # Sample every Nth pixel to avoid huge files
    step = max(1, min(height, width) // 100)  # Max 10,000 points
    
    rows, cols = np.meshgrid(
        np.arange(0, height, step),
        np.arange(0, width, step),
        indexing='ij'
    )
    
    rows = rows.flatten()
    cols = cols.flatten()
    values = data_array[rows, cols]
    
    # Filter valid values
    valid_mask = np.isfinite(values)
    rows = rows[valid_mask]
    cols = cols[valid_mask]
    values = values[valid_mask]
    
    df = pd.DataFrame({
        'X': cols,
        'Y': rows,
        'Value': np.round(values, 4)
    })
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Index_Values')
    
    return len(df)

@app.route('/api/analyze-bands', methods=['POST'])
def analyze_bands():
    if not verify_token():
        return jsonify({'error': 'Unauthorized'}), 401
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    temp_path = UPLOAD_FOLDER / f"input_{timestamp}.tif"
    
    try:
        cleanup_old_files()
        file.save(temp_path)
        band_info = analyze_tiff_basic(str(temp_path))
        temp_path.unlink()
        return jsonify(band_info)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        return jsonify({'error': str(e)}), 500

@app.route('/api/calculate-index', methods=['POST'])
def calculate_index_endpoint():
    if not verify_token():
        return jsonify({'error': 'Unauthorized'}), 401
    
    if 'file' not in request.files or 'index' not in request.form:
        return jsonify({'error': 'Missing file or index parameter'}), 400
    
    file = request.files['file']
    index_name = request.form['index'].upper()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    input_path = UPLOAD_FOLDER / f"input_{timestamp}.tif"
    
    try:
        cleanup_old_files()
        file.save(input_path)
        
        # Calculate index
        result_array = calculate_simple_index(str(input_path), index_name)
        
        # Create Excel
        output_excel = str(input_path).replace('.tif', f'_{index_name}.xlsx')
        point_count = create_points_excel(result_array, output_excel)
        
        response = send_file(
            output_excel,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'{index_name}_results.xlsx'
        )
        response.headers['X-Point-Count'] = str(point_count)
        
        @response.call_on_close
        def cleanup():
            try:
                input_path.unlink()
                Path(output_excel).unlink()
            except:
                pass
        
        return response
    except Exception as e:
        if input_path.exists():
            input_path.unlink()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    cleanup_old_files()
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'mode': 'basic_mode'
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'service': 'Satellite Processing API',
        'version': '1.0',
        'mode': 'basic'
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

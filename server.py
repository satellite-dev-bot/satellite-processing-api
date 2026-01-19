from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import rasterio
import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime
import tempfile

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

def get_band_info(raster_path):
    with rasterio.open(raster_path) as src:
        band_info = {
            'band_count': src.count,
            'bands': [],
            'width': src.width,
            'height': src.height,
            'crs': str(src.crs) if src.crs else 'Unknown',
            'bounds': list(src.bounds),
            'file_size': os.path.getsize(raster_path)
        }
        
        for i in range(1, src.count + 1):
            try:
                band = src.read(i)
                nodata = src.nodatavals[i-1] if src.nodatavals else None
                
                if nodata is not None:
                    band_masked = np.ma.masked_equal(band, nodata)
                else:
                    band_masked = np.ma.masked_invalid(band)
                
                band_meta = {
                    'band_number': i,
                    'name': src.descriptions[i-1] if src.descriptions and src.descriptions[i-1] else f'Band_{i}',
                    'data_type': str(src.dtypes[i-1]),
                    'min_value': float(band_masked.min()) if band_masked.count() > 0 else None,
                    'max_value': float(band_masked.max()) if band_masked.count() > 0 else None,
                    'mean_value': float(band_masked.mean()) if band_masked.count() > 0 else None,
                    'nodata': nodata,
                    'resolution': src.res
                }
                
                band_info['bands'].append(band_meta)
            except Exception as e:
                print(f"Error reading band {i}: {e}")
                continue
        
        return band_info

def calculate_index(raster_path, index_name):
    index_formulas = {
        'NDVI': lambda nir, red: (nir - red) / (nir + red + 1e-10),
        'NDRE': lambda nir, rededge: (nir - rededge) / (nir + rededge + 1e-10),
        'EVI': lambda nir, red, blue: 2.5 * ((nir - red) / (nir + 6*red - 7.5*blue + 1 + 1e-10)),
        'SAVI': lambda nir, red: ((nir - red) / (nir + red + 0.5 + 1e-10)) * 1.5,
        'NDWI': lambda green, nir: (green - nir) / (green + nir + 1e-10),
        'GNDVI': lambda nir, green: (nir - green) / (nir + green + 1e-10),
    }
    
    if index_name not in index_formulas:
        raise ValueError(f"Unsupported index: {index_name}")
    
    output_path = str(raster_path).replace('.tif', f'_{index_name}.tif')
    
    with rasterio.open(raster_path) as src:
        band_count = src.count
        
        if band_count == 3:
            bands = {'red': 1, 'green': 2, 'blue': 3}
        elif band_count == 4:
            bands = {'red': 1, 'green': 2, 'blue': 3, 'nir': 4}
        elif band_count >= 5:
            bands = {'blue': 1, 'green': 2, 'red': 3, 'nir': 4, 'rededge': 5}
        else:
            raise ValueError(f"Insufficient bands")
        
        if index_name == 'NDVI':
            nir = src.read(bands['nir']).astype(float)
            red = src.read(bands['red']).astype(float)
            result = index_formulas['NDVI'](nir, red)
        elif index_name == 'NDRE':
            nir = src.read(bands['nir']).astype(float)
            rededge = src.read(bands['rededge']).astype(float)
            result = index_formulas['NDRE'](nir, rededge)
        elif index_name == 'EVI':
            nir = src.read(bands['nir']).astype(float)
            red = src.read(bands['red']).astype(float)
            blue = src.read(bands['blue']).astype(float)
            result = index_formulas['EVI'](nir, red, blue)
        elif index_name == 'SAVI':
            nir = src.read(bands['nir']).astype(float)
            red = src.read(bands['red']).astype(float)
            result = index_formulas['SAVI'](nir, red)
        elif index_name == 'NDWI':
            green = src.read(bands['green']).astype(float)
            nir = src.read(bands['nir']).astype(float)
            result = index_formulas['NDWI'](green, nir)
        elif index_name == 'GNDVI':
            nir = src.read(bands['nir']).astype(float)
            green = src.read(bands['green']).astype(float)
            result = index_formulas['GNDVI'](nir, green)
        
        nodata_value = -9999
        if src.nodata is not None:
            result[nir == src.nodata] = nodata_value
        
        result = np.clip(result, -1, 1)
        result[~np.isfinite(result)] = nodata_value
        
        profile = src.profile.copy()
        profile.update(dtype=rasterio.float32, count=1, nodata=nodata_value)
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(result.astype(rasterio.float32), 1)
    
    return output_path

def raster_to_points_excel(raster_path, output_excel):
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        transform = src.transform
        nodata = src.nodata if src.nodata is not None else -9999
        
        valid_mask = (data != nodata) & np.isfinite(data)
        rows, cols = np.where(valid_mask)
        
        if len(rows) == 0:
            raise ValueError("No valid data")
        
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        values = data[rows, cols]
        
        df = pd.DataFrame({
            'X': [round(x, 6) for x in xs],
            'Y': [round(y, 6) for y in ys],
            'Value': [round(v, 4) for v in values]
        })
        
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Index_Values')
        
        return len(df)

@app.route('/api/analyze-bands', methods=['POST'])
def analyze_bands():
    if not verify_token():
        return jsonify({'error': 'Unauthorized'}), 401
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    temp_path = UPLOAD_FOLDER / f"input_{timestamp}.tif"
    
    try:
        cleanup_old_files()
        file.save(temp_path)
        band_info = get_band_info(str(temp_path))
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
        return jsonify({'error': 'Missing params'}), 400
    
    file = request.files['file']
    index_name = request.form['index'].upper()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    input_path = UPLOAD_FOLDER / f"input_{timestamp}.tif"
    
    try:
        cleanup_old_files()
        file.save(input_path)
        index_raster = calculate_index(str(input_path), index_name)
        output_excel = str(input_path).replace('.tif', f'_{index_name}.xlsx')
        point_count = raster_to_points_excel(index_raster, output_excel)
        
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
                Path(index_raster).unlink()
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
        'timestamp': datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'service': 'Satellite Processing API',
        'version': '1.0'
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
gunicorn==21.2.0

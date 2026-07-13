from flask import Flask, render_template, request, jsonify
from absa_predictor import create_predictor
import traceback

app = Flask(__name__)

predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = create_predictor()
    return predictor

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Teks opini tidak boleh kosong'}), 400
        
        pred = get_predictor()
        results = pred.predict(text)
        
        response = {
            'input_text': text,
            'results': results,
            'total_aspects': len(results)
        }
        
        return jsonify(response)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000)
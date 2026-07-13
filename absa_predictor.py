import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import re

class ABSAPredictor:
    def __init__(self, model_path: str = "model_indobert_absa"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.aspects = {
            'logistik_bantuan': {
                'name': 'Logistik dan bantuan',
                'keywords': ['logistik', 'bantuan', 'sembako', 'beras', 'mie instan', 'air bersih', 'makanan', 'pakaian', 'tenda', 'selimut', 'dapur umum', 'donasi', 'distribusi', 'penyaluran', 'posko bantuan']
            },
            'evakuasi_pengungsian': {
                'name': 'Evakuasi dan pengungsian',
                'keywords': ['evakuasi', 'mengungsi', 'pengungsian', 'pengungsi', 'tenda pengungsi', 'korban', 'relokasi', 'penyelamatan', 'sar', 'perahu', 'akses evakuasi', 'warga terdampak']
            },
            'koordinasi_pemerintah': {
                'name': 'Koordinasi pemerintah',
                'keywords': ['pemerintah', 'pemda', 'bpbd', 'bnpb', 'dinsos', 'dinas sosial', 'gubernur', 'bupati', 'camat', 'koordinasi', 'komando', 'tanggap darurat', 'posko', 'kebijakan', 'aparat', 'bantuan pemerintah']
            },
            'infrastruktur_rekonstruksi': {
                'name': 'Infrastruktur dan rekonstruksi',
                'keywords': ['infrastruktur', 'rekonstruksi', 'rehabilitasi', 'jalan', 'jembatan', 'listrik', 'air', 'jaringan', 'sinyal', 'sekolah', 'rumah', 'perbaikan', 'tanggul', 'drainase', 'banjir', 'longsor', 'irigasi']
            },
            'layanan_kesehatan': {
                'name': 'Layanan kesehatan',
                'keywords': ['kesehatan', 'puskesmas', 'rumah sakit', 'dokter', 'obat', 'medis', 'sanitasi', 'penyakit', 'diare', 'trauma', 'pos kesehatan', 'layanan kesehatan', 'ambulans', 'imunisasi']
            }
        }
        
        self.id2label = {0: 'negatif', 1: 'netral', 2: 'positif'}
        self.label2id = {'negatif': 0, 'netral': 1, 'positif': 2}
        
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def detect_aspects(self, text: str) -> List[str]:
        detected = []
        text_lower = text.lower()
        for aspect_slug, aspect_info in self.aspects.items():
            for keyword in aspect_info['keywords']:
                if keyword.lower() in text_lower:
                    detected.append(aspect_slug)
                    break
        return detected if detected else list(self.aspects.keys())
    
    def predict_sentiment(self, text: str, aspect: str) -> Dict[str, Any]:
        aspect_name = self.aspects[aspect]['name']
        
        inputs = self.tokenizer(
            text,
            aspect_name,
            truncation=True,
            max_length=160,
            padding='max_length',
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        
        pred_id = np.argmax(probs)
        confidence = float(probs[pred_id])
        
        return {
            'aspect_slug': aspect,
            'aspect_name': aspect_name,
            'sentimen_prediksi': self.id2label[pred_id],
            'prob_negatif': float(probs[0]),
            'prob_netral': float(probs[1]),
            'prob_positif': float(probs[2]),
            'confidence': confidence
        }
    
    def predict(self, text: str) -> List[Dict[str, Any]]:
        detected_aspects = self.detect_aspects(text)
        results = []
        
        for aspect in detected_aspects:
            result = self.predict_sentiment(text, aspect)
            result['aspek_terdeteksi_keyword'] = aspect in self.detect_aspects(text)
            results.append(result)
        
        return results


def create_predictor(model_path: str = "model_indobert_absa") -> ABSAPredictor:
    return ABSAPredictor(model_path)
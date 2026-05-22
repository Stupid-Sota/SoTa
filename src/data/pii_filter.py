"""
PII (Personally Identifiable Information) Detection and Filtering.
Detects emails, phones, SSNs, credit cards, IPs, and more.
Implements: PII expert training data, rule-based + ML hybrid detection.
"""

import re
import json
import os
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class PIIDetection:
    """A single PII detection result."""
    type: str
    text: str
    start: int
    end: int
    confidence: float


@dataclass
class PIIDetectionResult:
    """Full PII detection result for a text."""
    detections: List[PIIDetection] = field(default_factory=list)
    filtered_text: str = ''
    has_pii: bool = False
    num_detections: int = 0


PII_PATTERNS = {
    'email': r'[\w.+-]+@[\w-]+\.[\w.-]+',
    'phone': r'\b\+?\d{1,3}[\s.-]?\d{3}[\s.-]?\d{3}[\s.-]?\d{4}\b',
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
    'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    'date_of_birth': r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
    'zip_code': r'\b\d{5}(?:-\d{4})?\b',
}

PII_LABELS = ['normal', 'pii', 'redacted']

MASK_TOKEN = '[REDACTED]'


class PIIProcessor:
    """
    PII detection and filtering using regex patterns.
    Supports multiple filtering modes plus token-level classification.
    """

    def __init__(self, patterns: Optional[Dict[str, str]] = None):
        self.patterns = patterns or PII_PATTERNS
        self.compiled = {
            name: re.compile(pattern)
            for name, pattern in self.patterns.items()
        }

    def detect(self, text: str) -> PIIDetectionResult:
        result = PIIDetectionResult()
        for pii_type, regex in self.compiled.items():
            for match in regex.finditer(text):
                detection = PIIDetection(
                    type=pii_type,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95,
                )
                result.detections.append(detection)
        result.has_pii = len(result.detections) > 0
        result.num_detections = len(result.detections)
        return result

    def filter(self, text: str, mode: str = 'mask') -> str:
        if mode == 'mask':
            for pii_type, regex in self.compiled.items():
                text = regex.sub(MASK_TOKEN, text)
        elif mode == 'remove':
            for pii_type, regex in self.compiled.items():
                text = regex.sub('', text)
        elif mode == 'tag':
            for pii_type, regex in self.compiled.items():
                text = regex.sub(f'<{pii_type}>\\g<0></{pii_type}>', text)
        return text

    def tokenize_and_classify(self, tokenizer, text: str,
                               return_tensors: str = 'pt',
                               max_length: int = 512) -> Dict:
        tokens = tokenizer(text, return_tensors=return_tensors,
                           max_length=max_length, truncation=True)
        if tokens['input_ids'].size(1) >= max_length:
            logging.getLogger('sota').warning(
                f"Truncated [pii/tokenize] from >={max_length} to {max_length} tokens")
        return {'input_ids': tokens['input_ids'],
                'attention_mask': tokens['attention_mask']}

    def generate_pii_data(self, tokenizer, num_samples: int = 200,
                          output_path: str = None) -> List[Dict]:
        samples = []
        pii_texts = self._generate_pii_texts()
        clean_texts = [
            "Hello, how are you doing today?",
            "The weather is nice and sunny.",
            "Chess is a fascinating game of strategy.",
            "I enjoy reading books about history.",
            "The meeting is scheduled for tomorrow at 3pm.",
            "Please find the attached document for your review.",
            "Thank you for your prompt response.",
            "The project deadline has been extended by two weeks.",
            "Let me know if you have any questions.",
            "Have a wonderful day!",
        ]

        for _ in range(num_samples):
            if random.random() < 0.4:
                text = random.choice(pii_texts)
                has_pii = True
            else:
                text = random.choice(clean_texts)
                has_pii = False

            detection = self.detect(text)
            filtered = self.filter(text)

            samples.append({
                'original': text,
                'filtered': filtered,
                'has_pii': has_pii,
                'num_pii': detection.num_detections,
                'pii_types': [d.type for d in detection.detections],
            })

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
            print(f"[PII] Saved {len(samples)} samples to {output_path}")
        return samples

    def _generate_pii_texts(self) -> List[str]:
        emails = ['user@example.com', 'john.doe@gmail.com',
                  'alice@company.org', 'support@service.net']
        phones = ['+1-555-123-4567', '555-987-6543', '+34 612 345 678',
                  '212-555-0198']
        people_data = [
            f"My email is {e} and my phone is {p}."
            for e in emails for p in phones[:2]
        ]
        ssn_texts = [
            "My SSN is 123-45-6789.",
            "Please verify ID: 987-65-4321.",
            "Tax ID: 456-78-9012.",
        ]
        ip_texts = [
            "Server IP: 192.168.1.1",
            "Access from 10.0.0.45",
            "Gateway: 172.16.0.1",
        ]
        return people_data + ssn_texts + ip_texts


import random

"""
Translation Data Pipeline.
Generates EN→ES, ES→EN, EN→FR, FR→EN pairs for fine-tuning.
Implements: #111-114 (Translation rewards), translation expert training data.
"""

import random
import json
import os
import logging
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset


LANGUAGE_PAIRS = [
    ('en', 'es'), ('es', 'en'),
    ('en', 'fr'), ('fr', 'en'),
    ('en', 'de'), ('de', 'en'),
    ('en', 'it'), ('it', 'en'),
    ('en', 'pt'), ('pt', 'en'),
]

PHRASE_TEMPLATES = [
    "Hello, how are you?",
    "What time is it?",
    "Where is the train station?",
    "I would like a coffee, please.",
    "Thank you very much.",
    "Good morning, how can I help you?",
    "The weather is nice today.",
    "Please write your name here.",
    "How much does this cost?",
    "I don't understand. Can you repeat that?",
    "Chess is a game of strategy.",
    "The knight moves in an L-shape.",
    "Checkmate in three moves.",
    "She plays chess very well.",
    "The queen is the most powerful piece.",
    "Artificial intelligence is fascinating.",
    "Machine learning transforms data into insights.",
    "The neural network learned to play chess.",
    "Deep learning requires large datasets.",
    "Natural language processing enables translation.",
    "Knowledge is power.",
    "Practice makes perfect.",
    "Every master was once a beginner.",
    "The only way to do great work is to love what you do.",
    "In the middle of difficulty lies opportunity.",
]

TRANSLATIONS = {
    ('en', 'es'): {
        "Hello, how are you?": "Hola, ¿cómo estás?",
        "What time is it?": "¿Qué hora es?",
        "Where is the train station?": "¿Dónde está la estación de tren?",
        "I would like a coffee, please.": "Quisiera un café, por favor.",
        "Thank you very much.": "Muchas gracias.",
        "Good morning, how can I help you?": "Buenos días, ¿cómo puedo ayudarle?",
        "The weather is nice today.": "Hace buen tiempo hoy.",
        "Please write your name here.": "Por favor, escriba su nombre aquí.",
        "How much does this cost?": "¿Cuánto cuesta esto?",
        "I don't understand. Can you repeat that?": "No entiendo. ¿Puede repetir eso?",
        "Chess is a game of strategy.": "El ajedrez es un juego de estrategia.",
        "The knight moves in an L-shape.": "El caballo se mueve en forma de L.",
        "Checkmate in three moves.": "Jaque mate en tres movimientos.",
        "She plays chess very well.": "Ella juega ajedrez muy bien.",
        "The queen is the most powerful piece.": "La dama es la pieza más poderosa.",
        "Artificial intelligence is fascinating.": "La inteligencia artificial es fascinante.",
        "Machine learning transforms data into insights.": "El aprendizaje automático transforma datos en conocimiento.",
        "The neural network learned to play chess.": "La red neuronal aprendió a jugar ajedrez.",
        "Deep learning requires large datasets.": "El aprendizaje profundo requiere grandes conjuntos de datos.",
        "Natural language processing enables translation.": "El procesamiento del lenguaje natural permite la traducción.",
    },
    ('en', 'fr'): {
        "Hello, how are you?": "Bonjour, comment allez-vous ?",
        "What time is it?": "Quelle heure est-il ?",
        "Where is the train station?": "Où est la gare ?",
        "I would like a coffee, please.": "Je voudrais un café, s'il vous plaît.",
        "Thank you very much.": "Merci beaucoup.",
        "Good morning, how can I help you?": "Bonjour, comment puis-je vous aider ?",
        "The weather is nice today.": "Il fait beau aujourd'hui.",
        "Please write your name here.": "Veuillez écrire votre nom ici.",
        "How much does this cost?": "Combien ça coûte ?",
        "I don't understand. Can you repeat that?": "Je ne comprends pas. Pouvez-vous répéter ?",
        "Chess is a game of strategy.": "Les échecs sont un jeu de stratégie.",
        "The knight moves in an L-shape.": "Le cavalier se déplace en forme de L.",
        "Checkmate in three moves.": "Échec et mat en trois coups.",
        "She plays chess very well.": "Elle joue très bien aux échecs.",
        "The queen is the most powerful piece.": "La dame est la pièce la plus puissante.",
    },
    ('en', 'de'): {
        "Hello, how are you?": "Hallo, wie geht es Ihnen?",
        "What time is it?": "Wie spät ist es?",
        "Where is the train station?": "Wo ist der Bahnhof?",
        "I would like a coffee, please.": "Ich hätte gerne einen Kaffee, bitte.",
        "Thank you very much.": "Vielen Dank.",
        "Good morning, how can I help you?": "Guten Morgen, wie kann ich Ihnen helfen?",
        "The weather is nice today.": "Das Wetter ist heute schön.",
        "Chess is a game of strategy.": "Schach ist ein Strategiespiel.",
        "The knight moves in an L-shape.": "Der Springer bewegt sich in L-Form.",
        "Checkmate in three moves.": "Schachmatt in drei Zügen.",
    },
    ('en', 'it'): {
        "Hello, how are you?": "Ciao, come stai?",
        "What time is it?": "Che ora è?",
        "Where is the train station?": "Dov'è la stazione ferroviaria?",
        "I would like a coffee, please.": "Vorrei un caffè, per favore.",
        "Thank you very much.": "Grazie mille.",
        "Good morning, how can I help you?": "Buongiorno, come posso aiutarla?",
        "The weather is nice today.": "Il tempo è bello oggi.",
        "Chess is a game of strategy.": "Gli scacchi sono un gioco di strategia.",
    },
    ('en', 'pt'): {
        "Hello, how are you?": "Olá, como você está?",
        "What time is it?": "Que horas são?",
        "Where is the train station?": "Onde fica a estação de trem?",
        "I would like a coffee, please.": "Eu gostaria de um café, por favor.",
        "Thank you very much.": "Muito obrigado.",
        "Good morning, how can I help you?": "Bom dia, como posso ajudar?",
        "The weather is nice today.": "O tempo está bom hoje.",
        "Chess is a game of strategy.": "O xadrez é um jogo de estratégia.",
    },
}

for (src, tgt), pairs in list(TRANSLATIONS.items()):
    reverse_pairs = {}
    for en_text, translated in pairs.items():
        reverse_pairs[translated] = en_text
    TRANSLATIONS[(tgt, src)] = reverse_pairs


class TranslationDataset(Dataset):
    """Dataset for translation pairs across multiple languages."""

    def __init__(self, tokenizer, max_length: int = 128,
                 language_pairs: Optional[List[Tuple[str, str]]] = None,
                 size: int = 500):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.language_pairs = language_pairs or LANGUAGE_PAIRS
        self.samples = self._generate_samples(size)

    def _generate_samples(self, size: int) -> List[Dict]:
        samples = []
        for _ in range(size):
            pair = random.choice(self.language_pairs)
            src_lang, tgt_lang = pair
            key = (src_lang, tgt_lang)
            if key not in TRANSLATIONS:
                continue
            src_text = random.choice(list(TRANSLATIONS[key].keys()))
            tgt_text = TRANSLATIONS[key][src_text]
            samples.append({
                'prompt': f"translate {src_lang} to {tgt_lang}: {src_text}",
                'target': tgt_text,
                'src_lang': src_lang,
                'tgt_lang': tgt_lang,
                'source_text': src_text,
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        inputs = self.tokenizer(
            sample['prompt'], max_length=self.max_length, truncation=True,
            padding=False, return_tensors='pt',
        )
        targets = self.tokenizer(
            sample['target'], max_length=self.max_length, truncation=True,
            padding=False, return_tensors='pt',
        )
        if inputs['input_ids'].size(1) >= self.max_length:
            logger = logging.getLogger('sota')
            logger.warning(f"Truncated [translate/prompt] from >={self.max_length} "
                          f"to {self.max_length} tokens")
        if targets['input_ids'].size(1) >= self.max_length:
            logger = logging.getLogger('sota')
            logger.warning(f"Truncated [translate/target] from >={self.max_length} "
                          f"to {self.max_length} tokens")
        labels = targets['input_ids'].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels,
            'task': 'translate',
        }


def generate_translation_data(tokenizer, num_samples: int = 500,
                               output_path: str = None) -> List[Dict]:
    dataset = TranslationDataset(tokenizer, size=num_samples)
    data = []
    for sample in dataset.samples:
        data.append(sample)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[Translation] Saved {len(data)} samples to {output_path}")
    return data

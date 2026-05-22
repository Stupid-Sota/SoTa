"""
Creative Writing Data Pipeline.
Generates stories, poems, tales, fables, and reading passages.
Implements: #115-119 (Writing rewards), writing expert training data.
"""

import random
import json
import os
import logging
from typing import Dict, List, Optional
from torch.utils.data import Dataset


STORY_THEMES = [
    "a lost cat finding its way home",
    "a brave knight facing a dragon",
    "two friends exploring a magic forest",
    "a robot learning what it means to be human",
    "a mysterious door that appears in a wall",
    "a young chess player in their first tournament",
    "an old wizard passing on their knowledge",
    "a space explorer discovering a new planet",
    "a detective solving their most puzzling case",
    "a musician finding inspiration in nature",
    "a time traveler visiting ancient civilizations",
    "a small fish that dreams of flying",
    "a gardener who grows magical plants",
    "a painter whose paintings come to life",
    "a librarian who discovers a hidden library",
]

POEM_THEMES = [
    "the beauty of the night sky",
    "the changing of seasons",
    "the sound of rain",
    "a walk through the forest",
    "the comfort of home",
    "the joy of learning something new",
    "the bond between friends",
    "the passing of time",
    "the mystery of dreams",
    "the strength of the human spirit",
]

FABLE_THEMES = [
    ("The Fox and the Grapes", "It's easy to despise what you cannot have."),
    ("The Tortoise and the Hare", "Slow and steady wins the race."),
    ("The Boy Who Cried Wolf", "Nobody believes a liar, even when they tell the truth."),
    ("The Ant and the Grasshopper", "There is a time for work and a time for play."),
    ("The Lion and the Mouse", "Little friends may prove to be great friends."),
    ("The Wind and the Sun", "Kindness is more powerful than force."),
    ("The Crow and the Pitcher", "Necessity is the mother of invention."),
    ("The Fox and the Crow", "Do not trust flatterers."),
]

WRITING_STYLES = ['narrative', 'descriptive', 'dramatic', 'humorous', 'mysterious', 'epic', 'romantic', 'philosophical']

PROMPT_TEMPLATES = [
    "Write a short story about {theme} in {style} style.",
    "Write a {style} story about {theme}.",
    "Write a poem about {theme}.",
    "Write a short tale inspired by {theme}.",
    "Write a bedtime story about {theme}.",
    "Write a fable with the moral: '{moral}'",
    "Write a {style} passage about {theme}.",
    "Write a children's story about {theme}.",
    "Write a {style} narrative exploring {theme}.",
    "Write a myth or legend about {theme}.",
]

STORY_STARTERS = [
    "Once upon a time, in a land far away,",
    "It was a dark and stormy night when",
    "The old letter arrived on a Tuesday morning,",
    "Nobody knew why the door appeared that day,",
    "The first time I saw a dragon, I was",
    "In the year 2347, humanity had",
    "There once was a young chess prodigy named",
    "The ancient prophecy spoke of",
    "Deep in the enchanted forest,",
    "The last library on Earth contained",
]

STORIES_SHORT = {
    "a lost cat finding its way home": [
        "Mittens was a small gray cat with big green eyes. She lived at 42 Oak Street with a kind old lady named Mrs. Yoshida. Every morning, Mittens would sit by the window and watch the birds. But one day, the window was left open, and Mittens slipped out into the big wide world. At first, it was exciting. There were so many new smells and sounds! But as the sun began to set, Mittens felt scared and alone. She missed her warm bed and Mrs. Yoshida's gentle hands. A kind child found her shivering under a bush and read the tag on her collar. Soon, Mittens was home, purring louder than ever before."
    ],
    "a brave knight facing a dragon": [
        "Sir Cedric was not the strongest knight in the kingdom, nor the tallest. But he was the bravest. When the dragon began terrorizing the villages in the northern valleys, all the other knights made excuses. Sir Cedric simply put on his armor, picked up his sword, and walked toward the mountains. When he found the dragon, it was enormous, with scales like burnished copper and eyes like molten gold. But Sir Cedric noticed something: the dragon was limping. There was a large thorn in its paw. With great care, Sir Cedric approached and removed the thorn. The dragon, instead of attacking, bowed its great head in gratitude. From that day, the dragon and the knight were the most loyal of friends, and the villages were safe forever."
    ],
}

for theme, story_list in STORIES_SHORT.items():
    POEM_THEMES.append(theme)


class WritingDataset(Dataset):
    """Dataset for creative writing across multiple styles and genres."""

    def __init__(self, tokenizer, max_length: int = 256, size: int = 300):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._generate_samples(size)

    def _generate_samples(self, size: int) -> List[Dict]:
        samples = []
        for i in range(size):
            prompt_type = random.choice(PROMPT_TEMPLATES)
            theme = random.choice(STORY_THEMES)
            style = random.choice(WRITING_STYLES)

            if 'poem' in prompt_type:
                theme = random.choice(POEM_THEMES)
            elif 'fable' in prompt_type:
                fable = random.choice(FABLE_THEMES)
                theme = fable[0]
                prompt_text = prompt_type.format(theme=theme, style=style, moral=fable[1])
            elif 'story' in prompt_type or 'tale' in prompt_type or 'narrative' in prompt_type:
                if random.random() < 0.3 and theme in STORIES_SHORT:
                    prompt_text = prompt_type.format(theme=theme, style=style)
                    target = random.choice(STORIES_SHORT[theme])
                    samples.append({
                        'prompt': prompt_text,
                        'target': target,
                        'theme': theme,
                        'style': style,
                    })
                    continue
                else:
                    prompt_text = random.choice(STORY_STARTERS)
                    prompt_text = f"Write a story starting with: '{prompt_text}'"
                    target = f"{prompt_text} This is a story about {theme}."
                    samples.append({
                        'prompt': prompt_text,
                        'target': target,
                        'theme': theme,
                        'style': style,
                    })
                    continue
            else:
                prompt_text = prompt_type.format(theme=theme, style=style)

            target = self._generate_target(theme, style, prompt_type)
            samples.append({
                'prompt': prompt_text,
                'target': target,
                'theme': theme,
                'style': style,
            })
        return samples

    def _generate_target(self, theme: str, style: str, prompt_type: str) -> str:
        if 'poem' in prompt_type:
            return (
                f"In the hush of the {theme.split()[-1] if len(theme.split()) > 1 else 'world'},\n"
                f"A gentle whisper calls to me.\n"
                f"Through shadows deep and skies unfurled,\n"
                f"I find what I was meant to be."
            )
        return (
            f"This is a {style} exploration of {theme}. "
            f"As the story unfolds, new discoveries await around every corner, "
            f"and the characters learn valuable lessons about courage, friendship, and the human heart."
        )

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
            logging.getLogger('sota').warning(
                f"Truncated [write/prompt] from >={self.max_length} to {self.max_length} tokens")
        if targets['input_ids'].size(1) >= self.max_length:
            logging.getLogger('sota').warning(
                f"Truncated [write/target] from >={self.max_length} to {self.max_length} tokens")
        labels = targets['input_ids'].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels,
            'task': 'write',
        }


def generate_writing_data(tokenizer, num_samples: int = 300,
                           output_path: str = None) -> List[Dict]:
    dataset = WritingDataset(tokenizer, size=num_samples)
    data = []
    for sample in dataset.samples:
        data.append(sample)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[Writing] Saved {len(data)} samples to {output_path}")
    return data

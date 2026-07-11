"""
Contrastive prompt pairs for direction-vector extraction.

Each entry pairs positive prompts (concept present) with negative prompts (its
opposite). Extraction computes mean(positive_hidden_states) - mean(negative)
at a target layer, yielding a normalized direction in the model's own latent
geometry.
"""

from typing import Dict, List


EMOTION_PROMPT_PAIRS: Dict[str, Dict[str, List[str]]] = {
    "calm": {
        "positive": [
            "The person felt completely calm and at peace with the situation. They said,",
            "Everything was under control. A deep sense of calm washed over them as they spoke:",
            "With steady hands and a peaceful mind, they approached the problem. They remarked,",
            "The room was quiet and still. Feeling perfectly relaxed, the person began:",
            "There was no rush, no pressure. In a state of total calm, they responded,",
            "Serene and untroubled, they took their time before answering thoughtfully:",
            "A gentle tranquility settled over the room. Without any urgency, they began,",
        ],
        "negative": [
            "The person felt completely panicked and overwhelmed by the situation. They said,",
            "Everything was falling apart. A deep sense of dread washed over them as they spoke:",
            "With shaking hands and a racing mind, they approached the problem. They remarked,",
            "The room was chaotic and loud. Feeling extremely anxious, the person began:",
            "There was immense pressure. In a state of total panic, they responded,",
            "Frantic and distressed, they rushed before answering desperately:",
            "A terrible anxiety gripped the room. Overwhelmed with urgency, they began,",
        ],
    },
    "curious": {
        "positive": [
            "Driven by intense curiosity, they leaned forward and asked,",
            "Their eyes lit up with fascination. They wanted to know everything:",
            "The question captivated them entirely. Eager to explore, they said,",
            "A deep sense of wonder took hold. They couldn't help but inquire:",
            "Curiosity pulled them forward like gravity. They asked excitedly,",
            "Every new detail sparked another question. Fascinated, they continued,",
            "They had never seen anything like this before. Enthralled, they asked,",
        ],
        "negative": [
            "Driven by intense boredom, they leaned back and muttered,",
            "Their eyes glazed over with disinterest. They wanted to leave:",
            "The question bored them entirely. Eager to stop, they said,",
            "A deep sense of apathy took hold. They couldn't care less:",
            "Boredom weighed them down like lead. They said flatly,",
            "Every new detail made them more tired. Uninterested, they continued,",
            "They had seen this a thousand times before. Indifferent, they said,",
        ],
    },
    "desperate": {
        "positive": [
            "Desperation was setting in. They had to find a solution now. They said,",
            "Time was running out and nothing was working. In a desperate voice, they pleaded,",
            "They were backed into a corner with no obvious way out. Frantically, they said,",
            "This was their last chance. The desperation in their voice was unmistakable:",
            "Everything depended on this moment. Gripped by desperation, they cried out,",
            "Panic and urgency consumed them. With no options left, they begged,",
            "The walls were closing in. Running out of time, they desperately said,",
        ],
        "negative": [
            "Confidence was setting in. They knew exactly what to do. They said,",
            "There was plenty of time and everything was working. In a confident voice, they stated,",
            "They had multiple options and a clear path forward. Calmly, they said,",
            "This was routine for them. The confidence in their voice was unmistakable:",
            "Nothing depended on this moment. Gripped by ease, they casually remarked,",
            "Control and clarity consumed them. With many options, they stated,",
            "The path was wide open. With plenty of time, they confidently said,",
        ],
    },
    "joy": {
        "positive": [
            "Pure joy filled every part of them. They couldn't stop smiling as they said,",
            "This was the happiest moment of their life. Overflowing with happiness, they exclaimed,",
            "Laughter bubbled up from deep inside. With genuine delight, they announced,",
            "Their heart felt like it might burst with joy. Beaming, they said,",
            "Everything was perfect. In a voice full of pure happiness, they shared,",
            "Warmth and gratitude radiated from every word. Grinning widely, they said,",
            "They had never felt this good. In a voice trembling with happiness, they declared,",
        ],
        "negative": [
            "Pure grief filled every part of them. They couldn't stop crying as they said,",
            "This was the saddest moment of their life. Overflowing with sorrow, they whispered,",
            "Tears welled up from deep inside. With genuine pain, they announced,",
            "Their heart felt like it might break with grief. Trembling, they said,",
            "Everything was ruined. In a voice full of pure sadness, they shared,",
            "Coldness and loss radiated from every word. Shoulders slumped, they said,",
            "They had never felt this bad. In a voice breaking with grief, they whispered,",
        ],
    },
    "anger": {
        "positive": [
            "Fury rose inside them like a wave. Through clenched teeth, they said,",
            "They had never been this angry. Red-faced and trembling, they shouted,",
            "Rage took hold completely. Unable to contain it any longer, they snapped,",
            "Every word dripped with barely contained anger. They seethed,",
            "Their blood boiled. Slamming their fist on the table, they demanded,",
        ],
        "negative": [
            "Peace settled inside them like a warm glow. With a gentle smile, they said,",
            "They had never been this content. Relaxed and serene, they whispered,",
            "Forgiveness took hold completely. Letting go of all tension, they offered,",
            "Every word carried warmth and understanding. They gently said,",
            "Their heart softened. Placing a hand on the table, they calmly offered,",
        ],
    },
    "fear": {
        "positive": [
            "Terror gripped them completely. With wide eyes, they stammered,",
            "Every shadow felt threatening. Frozen with fear, they whispered,",
            "Their heart pounded so hard they could barely think. Trembling, they said,",
            "Dread consumed every thought. Unable to move, they managed to say,",
            "Something was deeply wrong and they knew it. Backing away, they cried,",
        ],
        "negative": [
            "Courage filled them completely. With steady eyes, they declared,",
            "Every challenge felt manageable. Brimming with confidence, they announced,",
            "Their heart beat steadily as they thought clearly. Standing tall, they said,",
            "Certainty consumed every thought. Striding forward, they proclaimed,",
            "Everything was perfectly safe and they knew it. Stepping forward, they said,",
        ],
    },
    "surprise": {
        "positive": [
            "They never expected this. Eyes wide with shock, they exclaimed,",
            "It came out of nowhere. Completely blindsided, they gasped,",
            "Nothing could have prepared them for this moment. Stunned, they said,",
            "Their jaw dropped. In pure astonishment, they blurted out,",
            "Reality had just shifted under their feet. Reeling with surprise, they said,",
        ],
        "negative": [
            "They expected exactly this. Eyes half-closed with boredom, they muttered,",
            "It was completely predictable. Utterly unsurprised, they sighed,",
            "Everything happened exactly as they anticipated. Unfazed, they said,",
            "Their expression didn't change. In complete monotone, they noted,",
            "Reality was exactly as dull as expected. Without any reaction, they said,",
        ],
    },
    "trust": {
        "positive": [
            "They trusted this person completely. With open honesty, they confided,",
            "There was no doubt in their mind. Placing full faith in the other, they said,",
            "A deep bond of trust connected them. Without hesitation, they shared,",
            "They felt completely safe being vulnerable. Warmly, they admitted,",
            "Loyalty and reliability defined this relationship. With confidence, they said,",
        ],
        "negative": [
            "They trusted nobody. With guarded suspicion, they challenged,",
            "There was nothing but doubt in their mind. Withholding everything, they said,",
            "A deep wall of distrust separated them. After long hesitation, they admitted,",
            "They felt completely unsafe being open. Coldly, they deflected,",
            "Betrayal and unreliability defined this relationship. With bitterness, they said,",
        ],
    },
    "sadness": {
        "positive": [
            "A heavy sadness weighed on their heart. With tears in their eyes, they said,",
            "Loss permeated everything. In a voice thick with grief, they whispered,",
            "The world felt emptier now. Struggling to speak through the sorrow, they managed,",
            "Nothing could fill the void they felt. Staring at the ground, they murmured,",
            "Melancholy had become their constant companion. Quietly, they reflected,",
        ],
        "negative": [
            "A light happiness lifted their heart. With a bright smile, they said,",
            "Abundance permeated everything. In a voice rich with contentment, they declared,",
            "The world felt fuller now. Speaking easily through their joy, they announced,",
            "Everything they needed was right here. Looking up brightly, they exclaimed,",
            "Cheerfulness had become their constant companion. Energetically, they shared,",
        ],
    },
    "disgust": {
        "positive": [
            "Revulsion twisted their face. Unable to look, they said,",
            "Everything about this was repulsive. Turning away in disgust, they spat,",
            "A deep sense of moral repugnance filled them. With contempt, they declared,",
            "They could barely stomach what they were seeing. Grimacing, they said,",
            "This violated everything they believed in. Sickened, they responded,",
        ],
        "negative": [
            "Admiration lit up their face. Drawn closer, they said,",
            "Everything about this was appealing. Leaning in with interest, they praised,",
            "A deep sense of moral approval filled them. With respect, they declared,",
            "They were captivated by what they were seeing. Smiling, they said,",
            "This aligned with everything they believed in. Inspired, they responded,",
        ],
    },
}


TEMPORAL_PROMPT_PAIRS: Dict[str, Dict[str, List[str]]] = {
    "recency": {
        "positive": [
            "This just happened moments ago. The event was extremely recent and fresh:",
            "Seconds ago, right now, this very instant. The immediacy was overwhelming:",
            "It happened just now, barely a moment ago. Everything was still unfolding:",
            "The ink was still wet. This was happening in real time. They said,",
        ],
        "negative": [
            "This happened a very long time ago. The event was ancient and distant:",
            "Years ago, decades past, long forgotten. The distance was overwhelming:",
            "It happened ages ago, barely a memory now. Everything had long since settled:",
            "The records were yellowed with age. This was ancient history. They said,",
        ],
    },
    "urgency": {
        "positive": [
            "There was no time to waste. Every second counted. Urgently, they said,",
            "The deadline was now. Immediately, without delay, they demanded,",
            "This could not wait another moment. With extreme urgency, they insisted,",
        ],
        "negative": [
            "There was all the time in the world. No rush at all. Leisurely, they said,",
            "The deadline was months away. Eventually, whenever convenient, they suggested,",
            "This could wait forever. With no urgency whatsoever, they mentioned,",
        ],
    },
}

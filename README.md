# EchoShield
## Introduction

This project aims to develop a reinforcement learning-based filter that confuses ASR models, like OpenAI’s Whisper, while remaining intelligible to humans. We will compare its effectiveness across multiple ASR systems and explore both static (e.g., band-pass filters) and dynamic (e.g., RL-based) defense mechanisms against such adversarial attacks. 


## RL Data Flow
### Attack
```mermaid
architecture-beta
    service audio(database)[audio]
    service model(abc)[RL Model] 
    service asr_normal(internet)[ASR] 
    service asr_attack(internet)[ASR]
    service text_normal(abc)[Normal Text]
    service text_attack(abc)[Garbled Text]
    service eval(abc)[Evaluate]
  
    junction a_m
    audio:B -- T:a_m
    a_m:R --> L:model
    model:R --> L:asr_attack
    asr_attack:R --> L:text_attack

    junction a_a
    audio:T -- B:a_a
    a_a:R --> L:asr_normal
    asr_normal:R --> L:text_normal
    junction extend_a_a
    text_normal:R -- L:extend_a_a

    junction text
    extend_a_a:B -- T:text
    text_attack:T -- B:text

    text:L --> R:eval
    eval:L --> T:model
```

The evaluation of the model will be done using the following metrics: 
- Minimal Change
- Text Similarit



## How to Use

### Prerequisites
You need python3 and an audio dataset. We choose a short but diverse audio dataset, found [here](https://github.com/tli725/JL-Corpus). You can download all the required data using the following commands:
``` bash
make dependencies
make data
```

### Build

TODO: Explain how the whole project can be build.

### Test

TODO: Explain how unit- or integreation tests can be executed.

### Run

TODO: Explain how to run the project (client, server etc.).

## Authors
- [Uplink036](https://github.com/Uplink036)
- [Adam Mützell](https://github.com/AdamMutzell)
### Contributions

The dataset was gathered from the following people, who asked you cite them as follows: "Jesin James, Li Tian, Catherine Watson, "An Open Source Emotional Speech Corpus for Human Robot Interaction Applications", in Proc. Interspeech, 2018."

## License

Copyright © 2025 Uplink036 
This work is licensed under [MIT]



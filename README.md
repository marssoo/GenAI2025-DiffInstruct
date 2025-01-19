# GenAI2025-DiffInstruct

**Grégoire MOURRE, Marceau LECLERC***

The file `Experiments.ipynb` recaps the commands to use to reproduce our experiments, and details the procedure we followed.

---

Organization of the repo:

```
.
├── Approach.ipynb
├── DiffInstruct.py
├── DI_models
│   ├── DI_generator.pth
│   ├── DI_phi.pth
│   └── tracking
│       └── logs_default.csv
├── DM_models
│   ├── UNet_4layers_128hc_2000steps_logs.csv
│   └── UNet_4layers_128hc_2000steps.pth
├── DM.py
├── eval_metrics.py
├── figures
│   ├── 5_di_round45.jpg
│   ├── 5_DM.jpg
│   ├── 5_g_pretrained.jpg
│   ├── 5_real.jpg.jpg
│   ├── DI_scores.jpg
│   ├── DI_training_logs.jpg
│   ├── DM_training_logs.jpg
│   └── viz_DI.jpg
├── GAN_models
│   ├── best_discriminator.pth
│   ├── best_generator.pth
│   └── GAN_logs.csv
├── GAN.py
├── README.md
├── train_DM.py
└── train_GAN.py

```


# k1_mjlab

```sh
uv run k1_list_envs
```

```sh
just tensorboard
```

```sh
just run k1_train Mjlab-Velocity-Rough-Booster-K1 \
    --env.scene.num-envs 4096 \
    --agent.max-iterations 30000 \
    --video True \
    --video-length 100 \
    --agent.logger tensorboard
```

```sh
just run k1_train Mjlab-Velocity-Flat-Booster-K1 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 30000 \
  --video True \
  --video-length 100 \
  --agent.logger tensorboard
```

```sh
just run k1_record Mjlab-Velocity-Rough-Booster-K1 \
  --checkpoint-file logs/rsl_rl/k1_velocity/[timestamp]/model_1950.pt \
  --num-envs 12 \
  --num-steps 1000
```

```sh
just run k1_play Mjlab-Velocity-Rough-Booster-K1 \
  --checkpoint-file [path-to-checkpoint] \
  --video True \
  --video-length 200

just run k1_play Mjlab-Velocity-Flat-Booster-K1 \
  --checkpoint-file [path-to-checkpoint] \
  --video True \
  --video-length 200
```

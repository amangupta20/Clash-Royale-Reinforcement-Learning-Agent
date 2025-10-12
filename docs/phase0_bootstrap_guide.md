# Phase 0 Bootstrap Training Guide

## Overview

This guide provides instructions for running the Phase 0 bootstrap training experiment with comprehensive TensorBoard logging and manual outcome input support.

## Features

- **Training Configuration**: 100K-500K steps with configurable PPO hyperparameters
- **Comprehensive TensorBoard Logging**: Action heatmaps, card selection patterns, elixir efficiency
- **Manual Outcome Input**: Support for manual win/loss input when automatic detection is not available
- **Training Continuation**: Ability to continue training from checkpoints
- **Performance Monitoring**: System resource tracking and benchmarking

## Quick Start

### 1. Basic Training

```bash
python -m src.bootstrap.train_bootstrap.py --config research/configs/phase0_bootstrap.yaml
```

### 2. Continue Training from Checkpoint

```bash
python -m src.bootstrap.train_bootstrap.py --config research/configs/phase0_bootstrap.yaml --continue-training --checkpoint ./artifacts/checkpoints/phase0_bootstrap/best_model_YYYYMMDD_HHMMSS
```

### 3. Custom Configuration

```bash
python -m src.bootstrap.train_bootstrap.py --config path/to/your/config.yaml
```

## Manual Outcome Input

During training, when automatic win/loss detection is not available, the system will prompt for manual input after 2 minutes of gameplay:

```
==================================================
MANUAL OUTCOME INPUT REQUIRED
==================================================
Please enter the game outcome:
  1 - Win
  2 - Loss
  3 - Continue playing
  0 - Skip/Cancel
==================================================
```

### Options:

- **1**: Records a win (+1.0 reward) and ends the episode
- **2**: Records a loss (-1.0 reward) and ends the episode
- **3**: Continues the current episode
- **0**: Skips manual outcome input

## Configuration

### Key Parameters

- `total_timesteps`: Initial training steps (default: 100,000)
- `target_win_rate`: Target win rate for Phase 0 (default: 0.35)
- `manual_outcome_input`: Enable manual outcome input (default: true)
- `outcome_check_delay_seconds`: Delay before prompting for outcome (default: 120)

### TensorBoard Logging

The system logs comprehensive metrics to TensorBoard:

#### Scalar Metrics

- Episode rewards (mean, min, max, std)
- Win rate progression
- Elixir efficiency
- Performance metrics

#### Visualizations

- Action heatmaps (32×18 grid)
- Card selection frequency
- Model weight distributions
- Gradient norms

#### Frequency Settings

- Heatmap logging: Every 5,000 steps
- Distribution logging: Every 2,000 steps
- Model logging: Every 10,000 steps

## Outputs

### Directory Structure

```
artifacts/checkpoints/phase0_bootstrap/
├── best_model_YYYYMMDD_HHMMSS
├── best_model_YYYYMMDD_HHMMSS_metadata.json
└── ppo_checkpoint_N_steps/

logs/phase0_bootstrap/
├── train.log
└── tensorboard_logs/

research/outputs/phase0_bootstrap/
├── training_summary.png
└── learning_curves/
```

### Key Files

- **Best Model**: `best_model_YYYYMMDD_HHMMSS.zip`
- **Model Metadata**: `best_model_YYYYMMDD_HHMMSS_metadata.json`
- **Training Log**: `logs/phase0_bootstrap/train.log`
- **TensorBoard Logs**: `logs/phase0_bootstrap/tensorboard_logs/`
- **Training Summary**: `research/outputs/phase0_bootstrap/training_summary.png`

## Monitoring Training

### TensorBoard

Launch TensorBoard to monitor training:

```bash
tensorboard --logdir logs/phase0_bootstrap
```

Access at: `http://localhost:6006`

### Key Metrics to Monitor

1. **Performance/WinRate**: Should trend toward 35% target
2. **Episodes/MeanReward**: Overall performance indicator
3. **Actions/Heatmap**: Shows deployment pattern evolution
4. **Cards/SelectionFrequency**: Card preference analysis
5. **Elixir/Efficiency**: Resource usage optimization

## Training Tips

### Phase 0 Success Criteria

- Target: 30-40% win rate vs easy AI
- Training: 100K steps initially
- Extend to 500K steps if target not achieved

### Manual Input Best Practices

1. **Be Consistent**: Use the same criteria for wins/losses
2. **Timing**: Input outcomes when games clearly end
3. **Continue Option**: Use "continue" for ambiguous games
4. **Skip Option**: Use "skip" when unsure

### Performance Optimization

- Monitor step times (target: <500ms P95)
- Check action heatmaps for deployment patterns
- Track elixir efficiency trends
- Watch for training divergence

## Troubleshooting

### Common Issues

1. **Module Import Errors**

   ```bash
   # Ensure you're in the project root
   cd /c/Clash-Royale-Reinforcement-Learning-Agent
   python -m src.bootstrap.train_bootstrap.py --config research/configs/phase0_bootstrap.yaml
   ```

2. **Directory Creation Errors**

   - Check permissions for artifacts/ and logs/ directories
   - Ensure sufficient disk space

3. **Manual Input Not Working**

   - Verify `manual_outcome_input: true` in config
   - Check that training has run for at least 2 minutes

4. **TensorBoard Not Showing Data**
   - Verify log directory exists
   - Check tensorboard_log_dir path in config
   - Restart TensorBoard server

### Performance Issues

1. **Slow Step Times**

   - Reduce resolution in capture settings
   - Optimize template matching thresholds
   - Check system resource usage

2. **Memory Issues**
   - Reduce batch size in PPO config
   - Decrease n_steps parameter
   - Monitor memory usage in TensorBoard

## Experiment Tracking

All experiments are logged to `research/experiments.jsonl` with:

- Git SHA for reproducibility
- Configuration hash
- Training results
- Performance metrics
- Model metadata

## Next Steps

After achieving Phase 0 targets:

1. Analyze TensorBoard logs for insights
2. Review action patterns and card preferences
3. Prepare for Phase 1 with enhanced features
4. Consider hyperparameter tuning for Phase 1

## Support

For issues or questions:

1. Check training logs: `logs/phase0_bootstrap/train.log`
2. Review TensorBoard metrics
3. Verify configuration syntax
4. Check system requirements

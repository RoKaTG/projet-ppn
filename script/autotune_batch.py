import subprocess
import optuna

def generate_topology(trial):
    num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 5)
    layers = [784]
    for i in range(num_hidden_layers):
        layer_size = trial.suggest_int(f'hidden_layer_{i}_size', 50, 600, step=25)
        layers.append(layer_size)
    layers.append(10)
    return ','.join(map(str, layers))

def objective(trial):
    # Hyper parameter definition
    topology = generate_topology(trial)
    activation = trial.suggest_categorical('activation', ['relu', 'sigmoid', 'fast_sigmoid', 'tanh', 'leaky', 'swish'])
    num_epochs = trial.suggest_int('num_epochs', 1, 10)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])

    # Build the command with the new batch_size argument
    command = f"./mlp true {topology} {activation} 60000 {num_epochs} {batch_size}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    output = result.stdout
    # Parse accuracy
    try:
        for line in output.split('\n'):
            if "Accuracy" in line:
                accuracy = float(line.split(':')[-1].strip().replace('%', ''))
                return accuracy
        # No accuracy = exception
        raise ValueError("Accuracy not found in the output.")
    except ValueError as e:
        print(f"Error parsing accuracy: {e}\nOutput was:\n{output}")
        return None  # Continue the tuning, just skip this iteration

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

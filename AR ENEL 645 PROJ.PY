import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Physical constants
mu = tf.cast(4 * np.pi * 1e-7, dtype=tf.float32)  # Magnetic permeability (H/m)
sigma_s = tf.constant(1.0, dtype=tf.float32)     # Magnetic charge density
epsilon = tf.constant(1e-5, dtype=tf.float32)    # Small constant for numerical stability

# Analytical solution for the magnetic flux leakage model
def analytical_flux_leakage(x, defect_length, defect_width, defect_depth):
    dz = tf.cast(defect_depth, dtype=tf.float32)
    dx = tf.cast(defect_length / 2, dtype=tf.float32)
    x = tf.cast(x, dtype=tf.float32)
    r1 = tf.sqrt((x - dx)**2 + dz**2 + epsilon)
    r2 = tf.sqrt((x + dx)**2 + dz**2 + epsilon)
    hz = (sigma_s / (4 * np.pi * mu)) * (1 / r1 - 1 / r2)
    return hz

# Generate training data along the centerline
def generate_centerline_data(num_points, range_limit, defect_params):
    x = np.linspace(-range_limit, range_limit, num_points, dtype=np.float32)
    hz = analytical_flux_leakage(x, defect_params["length"], defect_params["width"], defect_params["depth"])
    positions = np.stack([x / range_limit, np.zeros_like(x), np.zeros_like(x)], axis=1)
    return positions.astype(np.float32), hz.numpy()

# Define the Physics-Informed Neural Network (PINN)
class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden = [tf.keras.layers.Dense(32, activation="tanh", kernel_initializer='he_normal', dtype=tf.float32) for _ in range(4)]
        self.output_layer = tf.keras.layers.Dense(1, dtype=tf.float32)

    def call(self, inputs):
        x = tf.cast(inputs, dtype=tf.float32)
        for layer in self.hidden:
            x = layer(x)
        return self.output_layer(x)

# Boundary condition loss
def boundary_loss(model):
    boundary_points = tf.constant([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=tf.float32)  # Normalized boundaries
    boundary_values = model(boundary_points)
    return tf.reduce_mean(tf.square(boundary_values))  # Enforce H_z = 0 at boundaries

# Residual loss enforcing ∇²H_z = 0
def residual_loss(model, collocation_points):
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(collocation_points)
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(collocation_points)
            hz_pred = model(collocation_points)  # Predict flux leakage
        grads = tape2.gradient(hz_pred, collocation_points)  # First-order derivatives
    laplacian = tf.reduce_sum(tape1.gradient(grads, collocation_points), axis=1)  # Second-order derivatives (Laplacian)
    return tf.reduce_mean(tf.square(laplacian))  # Enforce ∇²H_z = 0

# Physics-based composite loss
def composite_loss(model, x_batch, y_batch, collocation_points):
    mse_loss = tf.reduce_mean(tf.square(model(x_batch) - y_batch))  # Prediction vs target
    bc_loss = boundary_loss(model)  # Boundary condition
    res_loss = residual_loss(model, collocation_points)  # PDE residual
    return mse_loss + bc_loss + res_loss  # Composite loss

# Training the PINN with composite loss
def train_pinn_centerline(pinn_model, train_dataset, collocation_points, epochs=1000):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0

        for x_batch, y_batch in train_dataset:
            with tf.GradientTape() as tape:
                loss = composite_loss(pinn_model, x_batch, y_batch, collocation_points)
            gradients = tape.gradient(loss, pinn_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, pinn_model.trainable_variables))
            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss.numpy())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss.numpy():.6f}")

    return loss_history

# Plotting function for flux leakage results
def plot_flux_leakage(axial_positions, predicted_hz, analytical_hz):
    plt.figure(figsize=(10, 6))
    plt.plot(axial_positions, predicted_hz, label="PINN Prediction", linestyle="--")
    plt.plot(axial_positions, analytical_hz, label="Analytical Solution", linestyle="-")
    plt.title("Normal Flux Leakage vs Axial Distance")
    plt.xlabel("Axial Distance (m)")
    plt.ylabel("Normal Magnetic Flux Leakage (Hz)")
    plt.legend()
    plt.grid()
    plt.show()

# Main execution block
if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Define defect parameters
    defect_params = {"length": 0.05, "width": 0.03, "depth": 0.02}
    num_points = 500
    range_limit = 0.1

    # Generate synthetic training data
    positions, hz_analytical = generate_centerline_data(num_points, range_limit, defect_params)
    hz_mean = np.mean(hz_analytical)
    hz_std = np.std(hz_analytical)
    hz_analytical_normalized = ((hz_analytical - hz_mean) / hz_std).astype(np.float32)

    # Initialize the PINN model
    pinn_model = PINN()

    # Create training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((positions, hz_analytical_normalized.reshape(-1, 1))).batch(32)

    # Generate collocation points for residual loss
    collocation_points = tf.random.uniform([500, 3], minval=-1.0, maxval=1.0, dtype=tf.float32)

    # Train the model
    loss_history = train_pinn_centerline(pinn_model, train_dataset, collocation_points, epochs=1000)

    # Generate predictions
    hz_predicted_normalized = pinn_model.predict(positions)
    hz_predicted = hz_predicted_normalized * hz_std + hz_mean  # Denormalize predictions

    # Plot the flux leakage comparison
    plot_flux_leakage(positions[:, 0] * range_limit, hz_predicted, hz_analytical)

    # Plot training loss history
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title("Training Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.grid()
    plt.show()

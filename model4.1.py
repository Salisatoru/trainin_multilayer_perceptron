import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ฟังก์ชันสำหรับสร้าง MLP และทำการฟอร์เวิร์ดพาส
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # กำหนด random weights สำหรับ hidden layer และ output layer
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.bias_hidden = np.random.uniform(-1, 1, (1, hidden_size))
        self.bias_output = np.random.uniform(-1, 1, (1, output_size))

    def forward(self, X):
        # การคำนวณแบบฟอร์เวิร์ดพาส
        self.hidden = self.sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        self.output = np.dot(self.hidden, self.weights_hidden_output) + self.bias_output
        return self.output

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def set_weights(self, weights):
        # แยก weights ตามโครงสร้าง MLP
        input_hidden_end = self.weights_input_hidden.size
        hidden_output_end = input_hidden_end + self.weights_hidden_output.size
        
        # reshape และ assign weights
        self.weights_input_hidden = weights[:input_hidden_end].reshape(self.weights_input_hidden.shape)
        self.weights_hidden_output = weights[input_hidden_end:hidden_output_end].reshape(self.weights_hidden_output.shape)
        self.bias_hidden = weights[hidden_output_end:hidden_output_end + self.bias_hidden.size].reshape(self.bias_hidden.shape)
        self.bias_output = weights[hidden_output_end + self.bias_hidden.size:].reshape(self.bias_output.shape)

# ฟังก์ชัน Objective สำหรับ PSO
def objective(weights, mlp, X_train, y_train, X_test, y_test):
    mlp.set_weights(weights)
    predictions = mlp.forward(X_test)
    mae = mean_absolute_error(y_test, predictions)
    return mae

# การติดตั้งและใช้งาน Particle Swarm Optimization (PSO)
class PSO:
    def __init__(self, num_particles, dimensions, bounds, max_iter, objective_func):
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.max_iter = max_iter
        self.objective_func = objective_func

        # เริ่มต้นตำแหน่งและความเร็ว
        self.positions = np.random.uniform(bounds[0], bounds[1], (num_particles, dimensions))
        self.velocities = np.random.uniform(-1, 1, (num_particles, dimensions))
        
        # ค่าเริ่มต้นของ p_best และ g_best
        self.p_best_positions = self.positions.copy()
        self.p_best_scores = np.full(num_particles, np.inf)
        self.g_best_position = None
        self.g_best_score = np.inf

    def optimize(self, mlp, X_train, y_train, X_test, y_test):
        mae_list = []
        
        for iteration in range(self.max_iter):
            for i in range(self.num_particles):
                score = self.objective_func(self.positions[i], mlp, X_train, y_train, X_test, y_test)
                
                # อัปเดต p_best
                if score < self.p_best_scores[i]:
                    self.p_best_scores[i] = score
                    self.p_best_positions[i] = self.positions[i].copy()
                    
                # อัปเดต g_best
                if score < self.g_best_score:
                    self.g_best_score = score
                    self.g_best_position = self.positions[i].copy()

            # อัปเดตตำแหน่งและความเร็ว
            w = 0.5  # inertia weight
            c1, c2 = 1.5, 1.5  # cognitive and social coefficients
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dimensions), np.random.rand(self.dimensions)
                cognitive_velocity = c1 * r1 * (self.p_best_positions[i] - self.positions[i])
                social_velocity = c2 * r2 * (self.g_best_position - self.positions[i])
                self.velocities[i] = w * self.velocities[i] + cognitive_velocity + social_velocity
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], self.bounds[0], self.bounds[1])

            mae_list.append(self.g_best_score)
            print(f'Iteration {iteration + 1}/{self.max_iter}, Best MAE: {self.g_best_score}')
        
        return self.g_best_position, mae_list

# โหลดข้อมูลและเตรียมข้อมูล
data = pd.read_excel("C:\c_programs/AirQualityUCI.xlsx", na_values=-200) # โหลด dataset จาก UCI repository
data.dropna(inplace=True)
X = data.iloc[:, [3, 6, 8, 10, 11, 12, 13, 14]].values # เลือกเฉพาะ attributes ที่ใช้เป็น input
y = data.iloc[:, 5].values # ค่าความเข้มข้น Benzene


# แบ่งชุดข้อมูลเป็น train และ test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# กำหนดโครงสร้าง MLP
input_size = X_train.shape[1]
hidden_size = 10
output_size = 1
mlp = MLP(input_size, hidden_size, output_size)

# กำหนดขอบเขตและจำนวน dimensions ของ PSO
dimensions = input_size * hidden_size + hidden_size * output_size + hidden_size + output_size
bounds = (-1, 1)
num_particles = 30
max_iter = 50

# เรียกใช้งาน PSO
pso = PSO(num_particles, dimensions, bounds, max_iter, objective)
optimal_weights, mae_list = pso.optimize(mlp, X_train, y_train, X_test, y_test)

# ตั้งค่า weights ที่ดีที่สุดและทำนายผล
mlp.set_weights(optimal_weights)
predictions = mlp.forward(X_test)


# แสดงกราฟของค่า MAE ในแต่ละรอบการฝึกและแสดงกราฟการเปรียบเทียบระหว่างค่าพยากรณ์และค่าจริง
plt.figure(figsize=(12, 6))
plt.plot(y, label="Actual", color="green", alpha=0.5)
plt.plot(y_pred, label="Predicted", color="orange", alpha=0.5)
for fold in range(1, num_folds):
    plt.axvline(x=fold * fold_size, color="black", linestyle="--", linewidth=1)
plt.xlabel("Sample Index")
plt.ylabel("Benzene Concentration (µg/m³)")
plt.title("Benzene Concentration: Actual vs Predicted")
plt.legend()
plt.grid()
plt.savefig(f"benzene_concentration_{hidden_layer_sizes[0]}_{average_mae}.png")
plt.show()




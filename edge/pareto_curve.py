import matplotlib.pyplot as plt

models    = ["TrOCR", "CRNN (ours)", "EfficientNetB0 (ours)", "MobileNetV3 (ours)"]
latency   = [820,     6.02,          0.03,                     0.03]
map_score = [0.7316,  0.4957,        0.7310,                   0.7438]
memory_mb = [1400,    55.53,         12.47,                     12.47]

plt.figure(figsize=(10, 6))

# Plot each point
for i, model in enumerate(models):
    # Scale HOG+LBP memory visually slightly less so it doesn't cover the whole plot, or use fixed scaling
    s_size = memory_mb[i] * 2 if memory_mb[i] < 1000 else memory_mb[i] / 2
    plt.scatter(latency[i], map_score[i], label=model, s=s_size, alpha=0.7)
    plt.annotate(model, (latency[i], map_score[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.xscale('log')
plt.xlabel('Latency (ms) [Log Scale]')
plt.ylabel('mAP@10')
plt.title('Performance vs Latency Pareto Curve')
plt.grid(True, which="both", ls="--", alpha=0.5)

# Save the plot
plt.savefig('edge/pareto_curve.png', bbox_inches='tight', dpi=300)
print("Saved pareto curve to edge/pareto_curve.png")

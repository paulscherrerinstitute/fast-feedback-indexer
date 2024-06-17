import numpy as np
import ffbidx

_60 = np.pi/3.0

Coords = np.reshape(
  np.array(
    [1.0, 4.0, 1.0, 2.0, 6.0, 4.0, 7.0, 8.0, 5.0, 9.0,
     2.0, 5.0, 1.0, 1.0, 3.0, 5.0, 1.0, 2.0, 7.0, 8.0,
     3.0, 6.0, 7.0, 4.0, 1.0, 2.0, 3.0, 6.0, 5.0, 8.0],
    dtype='float32'
  ),
  (10,3), order='F'
)

B0 = np.reshape(
  np.array(
    [2.0, 0.0, 0.0,
     0.0, 1.0, 0.0,
     0.0, 0.0, 1.0],
    dtype='float32'
  ),
  (3,3), order='F'
)

R60z = np.reshape(
  np.array(
    [np.cos(_60), np.sin(_60), 0,
     -np.sin(_60), np.cos(_60), 0,
     0, 0, 1],
    dtype='float32'
  ),
  (3,3), order='F'
)

B = np.matmul(B0, R60z.T, order='F')
Binv = np.asfortranarray(np.linalg.inv(B))
Spots = np.matmul(Coords, Binv.T, order='F')

print("True Coords:\n", Coords)
print("B0:\n", B0)
print("R60z:\n", R60z)
print("B:\n", B)
print("Binv:\n", Binv)
print("Spots:\n", Spots)

indexer = ffbidx.Indexer(32, 1, 10, 32, True)
Cand, Score = indexer.run(Spots, B0,
                          method='ifssr',
                          n_output_cells=32)
del indexer
best = np.argmin(Score)
Cand = np.reshape(Cand.ravel(), (3*32,3), order='F')

print("Cand\n", Cand)
print("Score\n", Score)
print("Best:", best)

sf = Score[best]
Bf = Cand[3*best:3*best+3, 0:3]
viable = (sf < .001)

print("B found:\n", Bf)
print(f"Score: {sf}, Viable: {viable}")
print("Computed Coords:\n", Spots@Bf.T)

import numpy as np

class flux_helper:
	def __init__(self, filename):
		T, Q = [], []
		with open(filename,'r') as f:
			for s in f.readlines():
				t,q = s.split(';')
				T.append(float(t))
				Q.append(float(q))
		self.TQ = np.array([T,Q])
		#self.TQ.sort(axis=0)
		self.T_PER = self.TQ[0][-1]
		#input(f'{self.TQ}, {self.T_PER}')

	def flux(self, t):
		time = t - (t // self.T_PER) * self.T_PER
		for i in range(len(self.TQ[0])-1):
			if self.TQ[0][i] <= time and time <= self.TQ[0][i+1]:
				t0, t1 = self.TQ[0][i], self.TQ[0][i+1]
				q0, q1 = self.TQ[1][i], self.TQ[1][i+1]
				alpha = (time-t0)/(t1-t0)
				q = q0 + alpha*(q1-q0)
				# if abs(t - 0.05) < 1.0e-4 or abs(t - 0.15) < 1.0e-4 or abs(t - 0.1) < 1.0e-4:
				# 	input(f't {t} q {q}')
				return q
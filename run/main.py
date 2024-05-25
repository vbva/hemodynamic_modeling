import numpy as np
import json
import pandas as pd
import sys
import matplotlib.pyplot as plt
from math import sqrt, pi, sin, exp
from scipy.linalg import lu_factor, lu_solve
import time
from multiprocessing.dummy import Pool as ThreadPool
from flux_helper import flux_helper


params_map = {
	"T_FINAL": "total_time",
	"DX": "space_step",
	"DT": "time_step", 
	"RHO": "density",
	"MU": "viscosity",
	"ALASTRUEY_GAMMA": "alastruey_gamma",
	"SCHEME_GAMMA": "scheme_gamma",
	"L": "length",
	"DIAM": "diameter",
	#"DIAM_OUT": "diameter_out",
	"AREA_DIA": "area_diastolic",
	"P_DIA": "pressure_diastolic",
	"C_SPEED": "perturbation_speed",
	"H_WALL": "wall_thickness",
	"E_YOUNG": "young_modulus",
	"T_PER": "time_period",
	"BC_TYPE": "bc_type",
	"BC_OUT": "outlet_type",
	"V_ID": "vessel_index",
	"P_OUT": "outflow_pressure",
	"WK_R1": "characteristic_impedance",
	"WK_R2": "total_peripheral_resistance",
	"WK_C": "total_arterial_compliance",
	"FLUX_TYPE": "flux_type",
	"FLUX_FILENAME": "flux_filename",
	"FLUX_ID": "analytics_index",
	"STATS_ID": "stats_index",
}

class DictWithAttributeAccess(dict):
	def __getattr__(self, key):
		return self[params_map[key]] if key in params_map else self[key]
 
	def __setattr__(self, key, value):
		if key in params_map:
			self[params_map[key]] = DictWithAttributeAccess(value) if type(value) is dict else value
		else:
			self[key] = value

class Vessel(DictWithAttributeAccess):
	def __init__(self, data, i):
		self.C_SPEED = 0.0
		self.AREA_DIA = 0.0
		DictWithAttributeAccess.__init__(self, data['vessel'][i])
		self.ALASTRUEY_BETA = 4./3 * sqrt(pi) * self.E_YOUNG * self.H_WALL
		self.tinn = np.zeros(5)

	def post_setup(self, model):
		self.bc_types = ['','']
		self.ALASTRUEY_GAMMA = model.ALASTRUEY_GAMMA
		self.SCHEME_GAMMA = model.SCHEME_GAMMA
		self.bc_order = model.bc_order
		self.RHO = model.RHO
		self.MU = model.MU
		self.DX = model.DX
		self.DT = min(model.DT, self.CFLTimeStep(model.CFL))
		if self.DT < model.DT:
			model.DT = self.DT
		self.ori = 1
		self.N = int( self.L / self.DX)
		self.AREA_IN = pi*self.DIAM**2 / 4
		#self.AREA_OUT = pi*self.DIAM_OUT**2 / 4
		if self.AREA_DIA == 0.0:
			self.AREA_DIA = self.AREA_IN
		self.S = np.ones(self.N) * self.AREA_IN
		self.U = np.zeros(self.N)
		self.P = np.ones(self.N) * self.P_DIA
		self.FS = np.zeros(self.N)
		self.FU = np.zeros(self.N)
		self.W1S = np.zeros(self.N)
		self.W1U = np.zeros(self.N)

	def linear_area(self, ind):
		return self.AREA_DIA
		x = ind*1.0/self.L * self.DX
		diam = (1.-x/L)*DIAM + x/L * DIAM_OUT
		return pi*diam**2/4

	def get_area(self, pressure):
		return np.power(np.sqrt(self.AREA_DIA) + self.AREA_DIA/self.ALASTRUEY_BETA*(pressure - self.P_DIA), 2)

	def pressure(self, area):
		return self.P_DIA + self.ALASTRUEY_BETA/self.AREA_DIA*(np.sqrt(area) - sqrt(self.AREA_DIA))

	def pressure_deriv(self, area):
		return self.ALASTRUEY_BETA/(2*self.AREA_DIA) * 1.0 / np.sqrt(area)

	def pressure_deriv2(self, area):
		return -self.ALASTRUEY_BETA/(4*self.AREA_DIA) * 1.0 / (area * np.sqrt(area) )

	def perturbation_velocity_coef(self, area):
		return np.sqrt(1.0 / (self.RHO * area) * self.pressure_deriv(area) )

	def friction(self, area, velocity):
		return -2.0*pi*(2.0+self.ALASTRUEY_GAMMA)*self.MU / self.RHO * np.divide(velocity,area)

	def CFLTimeStep(self, CFL):
		maxvel = 0.0
		if self.C_SPEED > 0.0:
			maxvel = self.C_SPEED
		else:
			maxvel = sqrt(self.E_YOUNG/self.RHO)
		assert maxvel > 0.0, 'Fail to setup CFL time step, incorrect input parameters'
		return CFL*self.DX/maxvel

	def step_inner_points(self):
		t = np.zeros(shape=len(self.tinn)+1)
		t[0] = time.time()
		tau, h = self.DT, self.DX
		N = self.N
		ori = self.ori
		SCHEME_GAMMA = self.SCHEME_GAMMA
		self.FS = np.multiply(self.S,self.U)
		self.FU = np.power(self.U,2)/2 + self.pressure(self.S) / self.RHO
		t[1] = time.time()
		# prepare work array
		self.W1S[1:-1] = self.S[1:-1] - 0.5*tau/h * ori * (self.FS[2:] - self.FS[:-2])
		self.W1U[1:-1] = self.U[1:-1] - 0.5*tau/h * ori * (self.FU[2:] - self.FU[:-2])
		if self.bc_order == 1:
			self.W1S[0]  = self.S[0]  - tau/h * ori * (self.FS[1] - self.FS[0])
			self.W1U[0]  = self.U[0]  - tau/h * ori * (self.FU[1] - self.FU[0])
			self.W1S[-1] = self.S[-1] - tau/h * ori * (self.FS[-1] - self.FS[-2])
			self.W1U[-1] = self.U[-1] - tau/h * ori * (self.FU[-1] - self.FU[-2])
		else:
			self.W1S[0]  = self.S[0]  - 0.5*tau/h * ori * (3*self.FS[2] - 4*self.FS[1] + self.FS[0])
			self.W1U[0]  = self.U[0]  - 0.5*tau/h * ori * (3*self.FU[2] - 4*self.FU[1] + self.FU[0])
			self.W1S[-1] = self.S[-1] + 0.5*tau/h * ori * (3*self.FS[-3] - 4*self.FS[-2] + self.FS[-1])
			self.W1U[-1] = self.U[-1] + 0.5*tau/h * ori * (3*self.FU[-3] - 4*self.FU[-2] + self.FU[-1])
		t[2] = time.time()
		# don't slice reused arrays if it can be cached:
		Sp,S0,Sm = self.S[1+ori:N-1+ori],self.S[1:-1],self.S[1-ori:N-1-ori]
		Up,U0,Um = self.U[1+ori:N-1+ori],self.U[1:-1],self.U[1-ori:N-1-ori]
		# compute next time step values into work array 
		Sp12,Up12 = 0.5*(S0+Sp),0.5*(U0+Up)
		Sm12,Um12 = 0.5*(S0+Sm),0.5*(U0+Um)
		v_plus  = self.perturbation_velocity_coef(Sp12)
		v_minus = self.perturbation_velocity_coef(Sm12)
		sigma_plus  = tau/h * np.vstack([Up12 - v_plus*Sp12, Up12 + v_plus*Sp12])
		sigma_minus = tau/h * np.vstack([Um12 - v_minus*Sm12, Um12 + v_minus*Sm12])
		t[3] = time.time()
		# assert np.all(sigma_plus[0] < 0.) and np.all(sigma_minus[0] < 0.), 'invalid sign in characteristic'
		# assert np.all(sigma_plus[1] > 0.) and np.all(sigma_minus[1] > 0.), 'invalid sign in characteristic'
		# assert np.all(abs(sigma_plus) < 1.), 'possibly too big time step, CFL violation'
		# assert np.all(abs(sigma_minus) < 1.), 'possibly too big time step, CFL violation'
		b_plus  = 0.5*np.multiply(np.abs(sigma_plus),  1. + 5./19*(1.-SCHEME_GAMMA)*(1.-np.abs(sigma_plus )) )
		b_minus = 0.5*np.multiply(np.abs(sigma_minus), 1. + 5./19*(1.-SCHEME_GAMMA)*(1.-np.abs(sigma_minus)) )
		d_plus  = 6./19 * (1.-SCHEME_GAMMA) * np.multiply(sigma_plus,  (1./np.abs(sigma_plus)  - 1.))
		d_minus = 6./19 * (1.-SCHEME_GAMMA) * np.multiply(sigma_minus, (1./np.abs(sigma_minus) - 1.))
		t[4] = time.time()
		def mult_OMEGA_INV_C_OMEGA_V2(vk,c,area,vel):
			return np.vstack([\
				0.5*np.multiply(c[0]+c[1],area) + 0.5*np.multiply(np.divide(c[1]-c[0], vk), vel),\
				0.5*np.multiply(np.multiply(c[1]-c[0],vk),area) + 0.5*np.multiply(c[1]+c[0],vel) ])
		dWb1 = mult_OMEGA_INV_C_OMEGA_V2(v_plus, b_plus, \
			Sp - S0, Up - U0)
		dWb0 = mult_OMEGA_INV_C_OMEGA_V2(v_minus, b_minus, \
			S0 - Sm, U0 - Um)
		dWd1 = mult_OMEGA_INV_C_OMEGA_V2(v_plus, d_plus, \
			self.W1S[1+ori:N-1+ori] - Sp + self.W1S[1:-1] - S0, \
			self.W1U[1+ori:N-1+ori] - Up + self.W1U[1:-1] - U0)
		dWd0 = mult_OMEGA_INV_C_OMEGA_V2(v_minus, d_minus, \
			self.W1S[1-ori:N-1-ori] - Sm + self.W1S[1:-1] - S0, \
			self.W1U[1-ori:N-1-ori] - Um + self.W1U[1:-1] - U0)
		FRIC = tau*self.friction(S0, U0)
		self.S[1:-1] = self.W1S[1:-1] + dWb1[0] - dWb0[0] + dWd1[0] - dWd0[0]
		self.U[1:-1] = self.W1U[1:-1] + dWb1[1] - dWb0[1] + dWd1[1] - dWd0[1] + FRIC
		t[5] = time.time()
		for i in range(len(self.tinn)):
			self.tinn[i] += t[i+1]-t[i]

class BC(DictWithAttributeAccess):
	def __init__(self, data):
		DictWithAttributeAccess.__init__(self, data)
		if self.BC_TYPE == 'junction':
			self.indices = [int(s) for s in data['vessel_index'].split(';')]

	def post_setup(self, model):
		if self.BC_TYPE == 'inlet':
			self.BC_SIDE = 0
			model.vessels[self.V_ID].ori = 1
			model.vessels[self.V_ID].bc_types[0] = 'inlet'
			if self.FLUX_TYPE == 'file':
				self.FH = flux_helper(self.FLUX_FILENAME)

	def compute_bc_coefs(self, model, vessel):
		N, ori = vessel.N, vessel.ori
		if self.BC_SIDE == 0:
			i0, di, sign = 0, 1, ori
		else:
			i0, di, sign = N-1, -1, -ori
		s, u = [0.]*3, [0.]*3
		for i in range(3):
			s[i] = vessel.S[i0 + i*di]
			u[i] = vessel.U[i0 + i*di]
		if model.bc_order == 1:
			se, ue, coef = s[1], u[1], 1.0
		else:
			se, ue, coef = 2*s[1] - 0.5*s[2], 2*u[1] - 0.5*u[2], 1.5
		w = vessel.perturbation_velocity_coef(s[0])
		sigma = vessel.DT/vessel.DX * (u[0] - sign*w*s[0])

		alpha = sign*w
		beta = (w*(sigma*se - sign*s[0]) + (u[0] - sign*sigma*ue) - sign*vessel.DT*vessel.friction(s[0], u[0])) \
			/ (1.-sign*coef*sigma)
		return alpha, beta

	# WINDKESSEL BC
	def wk_newton_f(self, vessel, dt, s, s0, u0, a, b):
		return 1.0/dt * ( 
			s*(a*s+b) * ( dt*(self.WK_R1+self.WK_R2) + self.WK_C*self.WK_R2*self.WK_R1 )
			- dt*(vessel.pressure(s) - self.P_OUT)
			- self.WK_C*self.WK_R2*vessel.pressure_deriv(s) * (s - s0)
			- self.WK_C*self.WK_R2*self.WK_R1 * (s0*u0)
			)

	def wk_newton_dfds(self, vessel, dt, s, s0, u0, a, b):
		return  1.0/dt * (
			(2*a*s+b) * ( dt*(self.WK_R1+self.WK_R2) + self.WK_C*self.WK_R2*self.WK_R1 )
			- dt*vessel.pressure_deriv(s)
			- self.WK_C*self.WK_R2*( vessel.pressure_deriv2(s) * (s - s0) + vessel.pressure_deriv(s) )
			)

	def analytic_flux(self, t):
		if self.FLUX_ID == 0:
			return exp(-10000.0 * (t-0.05)*(t-0.05))
		elif self.FLUX_ID == 1:
			# flux in ml/s:
			T = self.T_PER
			return 10e5*(7.9853e-06\
				+2.6617e-05*sin(2*pi*t/T+0.29498)+2.3616e-05*sin(4*pi*t/T-1.1403)-1.9016e-05*sin(6*pi*t/T+0.40435)\
				-8.5899e-06*sin(8*pi*t/T-1.1892)-2.436e-06*sin(10*pi*t/T-1.4918)+1.4905e-06*sin(12*pi*t/T+1.0536)\
				+1.3581e-06*sin(14*pi*t/T-0.47666)-6.3031e-07*sin(16*pi*t/T+0.93768)-4.5335e-07*sin(18*pi*t/T-0.79472)\
				-4.5184e-07*sin(20*pi*t/T-1.4095)-5.6583e-07*sin(22*pi*t/T-1.3629)+4.9522e-07*sin(24*pi*t/T+0.52495)\
				+1.3049e-07*sin(26*pi*t/T-0.97261)-4.1072e-08*sin(28*pi*t/T-0.15685)-2.4182e-07*sin(30*pi*t/T-1.4052)\
				-6.6217e-08*sin(32*pi*t/T-1.3785)-1.5511e-07*sin(34*pi*t/T-1.2927)+2.2149e-07*sin(36*pi*t/T+0.68178)\
				+6.7621e-08*sin(38*pi*t/T-0.98825)+1.0973e-07*sin(40*pi*t/T+1.4327)-2.5559e-08*sin(42*pi*t/T-1.2372)\
				-3.5079e-08*sin(44*pi*t/T+0.2328))
		elif self.FLUX_ID == 2:
			T = self.T_PER
			return 1e6*(
				3.1199+7.7982*sin(2*pi*t/T+0.5769)+4.1228*sin(4*pi*t/T-0.8738)-1.0611*sin(6*pi*t/T+0.7240)+0.7605*sin(8*pi*t/T-0.6387)
				-0.9148*sin(10*pi*t/T+1.1598)+0.4924*sin(12*pi*t/T-1.0905)-0.5580*sin(14*pi*t/T+1.042)+0.3280*sin(16*pi*t/T-0.5570)
				-0.3941*sin(18*pi*t/T+1.2685)+0.2833*sin(20*pi*t/T+0.6702)+0.2272*sin(22*pi*t/T-1.4983)+0.2249*sin(24*pi*t/T+0.9924)
				+0.2589*sin(26*pi*t/T-1.5616)-0.1460*sin(28*pi*t/T-1.3106)+0.2141*sin(30*pi*t/T-1.1306)-0.1253*sin(32*pi*t/T+0.1552)
				+0.1321*sin(34*pi*t/T-1.5595)-0.1399*sin(36*pi*t/T+0.4223)-0.0324*sin(38*pi*t/T+0.7811)-0.1211*sin(40*pi*t/T+1.0729)
				)/1000/60
		elif self.FLUX_ID == 3:
			#t += 0.055;
			T = self.T_PER
			# shift = 0.0;//-3.796+0.0006082;
			# flux in ml/s:
			return 6.5+3.294*sin(2*pi*t/T-0.023974)+1.9262*sin(4*pi*t/T-1.1801)-1.4219*sin(6*pi*t/T+0.92701)
			-0.66627*sin(8*pi*t/T-0.24118)-0.33933*sin(10*pi*t/T-0.27471)-0.37914*sin(12*pi*t/T-1.0557)
			+0.22396*sin(14*pi*t/T+1.22)+0.1507*sin(16*pi*t/T+1.0984)+0.18735*sin(18*pi*t/T+0.067483)
			+0.038625*sin(20*pi*t/T+0.22262)+0.012643*sin(22*pi*t/T-0.10093)-0.0042453*sin(24*pi*t/T-1.1044)
			-0.012781*sin(26*pi*t/T-1.3739)+0.014805*sin(28*pi*t/T+1.2797)+0.012249*sin(30*pi*t/T+0.80827)
			+0.0076502*sin(32*pi*t/T+0.40757)+0.0030692*sin(34*pi*t/T+0.195)-0.0012271*sin(36*pi*t/T-1.1371)
			-0.0042581*sin(38*pi*t/T-0.92102)-0.0069785*sin(40*pi*t/T-1.2364)+0.0085652*sin(42*pi*t/T+1.4539)
			+0.0081881*sin(44*pi*t/T+0.89599)+0.0056549*sin(46*pi*t/T+0.17623)+0.0026358*sin(48*pi*t/T-1.3003)
			-0.0050868*sin(50*pi*t/T-0.011056)-0.0085829*sin(52*pi*t/T-0.86463)

	def data_flux(self, t):
		return self.FH.flux(t)

	def flux(self, t):
		if self.FLUX_TYPE == 'analytics':
			return self.analytic_flux(t)
		else:
			return self.data_flux(t)

	def compute_junction_bc_bernoulli(self, model):
		N = len(self.indices)
		JJ = np.empty(shape=(N,N))
		RESID = np.zeros(shape=N)
		alphas, betas = np.empty(shape=N), np.empty(shape=N)
		AREA = np.empty(shape=N)
		ID = [0]*N
		ORI = np.empty(shape=N)
		for i in range(N):
			j = self.indices[i]
			vsl = model.vessels[j]
			alphas[i], betas[i] = self.compute_bc_coefs(model, vsl)
			ID[i] = 0 if self.BC_SIDE == 0 else vsl.N-1
			ORI[i] = vsl.ori
			AREA[i] = vsl.S[ID[i]]
		# NONLINEAR LOOP, NEWTON METHOD
		rnorm0, rnorm = 0., 0.
		niters, niters_max = 0, 1000
		fail = False
		RHO = model.RHO
		while True:

			D0 = model.vessels[0].DIAM
			A0 = model.vessels[0].AREA_IN
			stenosis_degree = 0.75
			stenosis_length = 4.8
			#SD_2 = np.power(1 / (1 - stenosis_degree), 2)  # (Ao / As)^2
			SD_2 = stenosis_degree
			K_v = 32 * (stenosis_length / D0) * np.power((1/ (1 - stenosis_degree)),2)
			K_t = 1.52
			Delta_P = K_v*model.MU*AREA[0]*(alphas[0]*AREA[0]+betas[0])/(D0*A0) + K_t*model.RHO*AREA[0]*(alphas[0]*AREA[0]+betas[0])*\
					  abs(AREA[0]*(alphas[0]*AREA[0]+betas[0]))*np.power(1/(1-stenosis_degree) - 1, 2) / (2 * np.power(A0, 2))
			


			# COMPUTE RESIDUAL
			rnorm = 0.
			P_RHO = np.array([model.vessels[self.indices[i]].pressure(AREA[i]) for i in range(N)])
			RESID[1:] = P_RHO[0] - P_RHO[1:]
			RESID[1:] = RESID[1:] - Delta_P
			RESID[0] = (ORI*AREA*(alphas*AREA+betas)).sum()
			rnorm = sqrt(RESID.dot(RESID))
			if rnorm0 == 0.:
				rnorm0 = rnorm
			# input(f'iter {niters} rnorm {rnorm} rnorm0 {rnorm0} rnorm/rnorm0 {rnorm/rnorm0} RESID {RESID}')
			# FINISH SUCCESSFULLY, IF THE RESIDUAL IS SMALL
			if rnorm < 1.0e-9 or rnorm/rnorm0 < 1.0e-4:
				fail = False
				break
			# BREAK UNSUCCESSFULLY IF AMOUNT OF ITERATIONS IS BIG OR RESIDUAL IS BIG
			if niters > niters_max or rnorm/rnorm0 > 1.0e10 or rnorm > 1.0e12:
				fail = True
				break
			# COMPOSE JACOBIAN
			JJ[0] = ORI*(2*alphas*AREA+betas) #0 строка #Ai*ui = Ai(alphai*Ai + bettai) = 0
			# FILL ROW i
			for i in range(1,N):
				u1 = 2*alphas[0]*AREA[0] + betas[0] #dQ/dA
				u2 = 4*np.power(alphas[0], 2)*np.power(AREA[0],3) + 6*alphas[0]*betas[0]* np.power(AREA[0],2) + 2*AREA[0] * np.power(betas[0], 2) #d(Q^2)/dA
				C1 = (K_v * model.MU)/(A0 * D0)
				#C2 = K_t * model.RHO*np.power(1.0/(1.0 - stenosis_degree) - 1.0, 2)/(2.0 * A0 * A0)
				C2 = K_t * model.RHO*np.power(SD_2- 1.0, 2)/(2.0 * A0 * A0)
				#ddP_dA = -ORI[i]*u*(C1 + u*C2) #ПОДГОН УМНОЖИЛИ НА МИНУС ЗОЧЕМ?????????
				ddP_dA = C1*u1 + C2*u2
				JJ[i][0] =  model.vessels[self.indices[0]].pressure_deriv(AREA[0]) - ddP_dA
				# u1 = 2*alphas[i]*AREA[0] + betas[i] #dQ/dA
				# u2 = 4*np.power(alphas[i], 2)*np.power(AREA[i],3) + 6*alphas[i]*betas[i]* np.power(AREA[0],2) + 2*AREA[i] * np.power(betas[i], 2) #d(Q^2)/dA
				# C1 = (K_v * model.MU)/(A0 * D0)
				# #C2 = K_t * model.RHO*np.power(1.0/(1.0 - stenosis_degree) - 1.0, 2)/(2.0 * A0 * A0)
				# C2 = K_t * model.RHO*np.power(SD_2- 1.0, 2)/(2.0 * A0 * A0)
				# #ddP_dA = -ORI[i]*u*(C1 + u*C2) #ПОДГОН УМНОЖИЛИ НА МИНУС ЗОЧЕМ?????????
				# ddP_dA = -ORI[i] * (C1*u1 + C2*u2)
				#print("ddP-DA", ddP_dA)
				JJ[i][i] = -model.vessels[self.indices[i]].pressure_deriv(AREA[i])
			if np.any(np.isnan(RESID)) or np.any(np.isinf(RESID)):
				input(f'T {model.T} bad values: inf {np.any(np.isinf(RESID))} nan {np.any(np.isnan(RESID))} area {AREA}')
			# SOLVE SYSTEM
			X = lu_solve( lu_factor(JJ), RESID)
			xnorm = sqrt(X.dot(X))
			# UPDATE AREA
			AREA -= X
			assert np.all(AREA > 0.0), 'negative area'
			niters += 1
			if abs(xnorm) < 1.0e-15:
				fail = rnorm < 1.0e-9 or rnorm/rnorm0 < 1.0e-4
				break
		if not fail:
			for i in range(N):
				j = self.indices[i]
				model.vessels[j].S[ID[i]] = AREA[i]
				model.vessels[j].U[ID[i]] = alphas[i]*AREA[i]+betas[i]
		else:
			input(f'Fail to solve nonlinear system for junction BC; T {model.T} iters {niters}\
			 rnorm {rnorm} rnorm0 {rnorm0} rnorm/rnorm0 {rnorm/rnorm0} xnorm {xnorm}')

	def compute_bc(self, model):
		# JUNCTION
		if self.BC_TYPE == 'junction': # BERNOULLI			
			self.compute_junction_bc_bernoulli(model)
		else:
			vessel = model.vessels[self.V_ID]
			alpha, beta = self.compute_bc_coefs(model, vessel)
			i = 0 if self.BC_SIDE == 0 else vessel.N-1
			if self.BC_TYPE == 'inlet':
				q = self.flux(model.T + model.DT)
				vessel.S[i] = ( -beta + sqrt(beta*beta + 4*alpha*q) ) / (2 * alpha)
				# assert vessel.S[i] > 0.0, 'negative area'
				# assert abs(vessel.S[i]*vessel.U[i] - q) < 1.0e-9, 'fail to setup inflow BC'
			elif self.BC_TYPE == 'outlet':
				if self.BC_OUT == 'outflow_pressure':
					vessel.S[i] = vessel.get_area(self.P_OUT)
				elif self.BC_OUT == 'windkessel':
					# Newton method
					s0, s1, u0 = [vessel.S[i]][0], [vessel.S[i]][0], [vessel.U[i]][0]
					resid0 = self.wk_newton_f(vessel, model.DT, s1, s0, u0, alpha, beta)
					resid, ds = [resid0][0], 1.0
					iters, fail = 0, False
					while True:
						if not (abs(ds) > 1.0e-12 and abs(resid) > 1.0e-15 and abs(resid/resid0) > 1.0e-12):
							fail = False
							# assert s1 > 0.0, 'negative area'
							break
						ds = -self.wk_newton_f(vessel, model.DT, s1, s0, u0, alpha, beta) \
						/ self.wk_newton_dfds(vessel, model.DT, s1, s0, u0, alpha, beta)
						s1 += ds
						resid = self.wk_newton_f(vessel, model.DT, s1, s0, u0, alpha, beta)
						iters += 1
						if s1 < 0.0:
							fail = True
							break
					vessel.S[i] = s1
			vessel.U[i] = alpha*vessel.S[i] + beta

class FlowModel():
	def __init__(self, filename):
		with open(filename, 'r') as f:
			data = json.load(f)
		self.T = 0.0
		self.T_FINAL = data['total_time']
		self.RHO = data['density']
		self.MU = data['viscosity']
		self.DX = data['space_step']
		self.DT = data['time_step']
		self.bc_order = data['bc_order']
		self.SCHEME_GAMMA = data['scheme_gamma']
		self.ALASTRUEY_GAMMA = data['alastruey_gamma']
		self.CFL = data['CFL']
		self.vessels = []
		self.savefig = data['savefig'] if 'savefig' in data else 'out0.png'
		for i in data['vessel'].keys():
			self.vessels.append(Vessel(data, i))
			self.vessels[-1].post_setup(self)
		self.pool = ThreadPool(len(self.vessels))
		print(f'CFL {self.CFL} time step: {self.DT}')
		self.bcs = []
		for i in data['bc'].keys():
			self.bcs.append(BC(data['bc'][i]))
			self.bcs[-1].post_setup(self)
		# process junction BC and provide correct orientations
		mark = np.zeros(shape=len(self.bcs))
		for i in range(len(self.bcs)):
			if self.bcs[i].BC_TYPE == 'junction':
				mark[i] = 1
		while True:
			all_processed = True
			for i in range(len(self.bcs)):
				if mark[i]:
					all_processed = False
					# one of two vessel sides on which BC is not set yet
					k1_side = -1
					for j in range(len(self.bcs[i].indices)):
						k1 = self.bcs[i].indices[j]
						k1_ori = self.vessels[k1].ori
						if self.vessels[k1].bc_types[0] != '':
							k1_side = 1
						elif self.vessels[k1].bc_types[1] != '':
							k1_side = 0
						if k1_side != -1:
							break
					# indicates left (0) or right (1) side of the vessel with index bcs[i].V_ID
					self.bcs[i].BC_SIDE = k1_side
					if k1_side >= 0:
						for j in range(len(self.bcs[i].indices)):
							k2 = self.bcs[i].indices[j]
							# change orientation of all vessel except k1:
							self.vessels[k2].ori = k1_ori if k1 == k2 else -k1_ori
							# all vessel are juncted at this end
							self.vessels[k2].bc_types[k1_side] = 'junction'
						# junction BC is processed
						mark[i] = 0
			if all_processed:
				break
		# setup outlet BC on remaining unset vessel ends
		for i in range(len(self.bcs)):
			if self.bcs[i].BC_TYPE == 'outlet':
				if self.vessels[self.bcs[i].V_ID].bc_types[0] == '':
					self.bcs[i].BC_SIDE = 0
					self.vessels[self.bcs[i].V_ID].bc_types[0] = 'outlet'
				elif self.vessels[self.bcs[i].V_ID].bc_types[1] == '':
					self.bcs[i].BC_SIDE = 1
					self.vessels[self.bcs[i].V_ID].bc_types[1] = 'outlet'
				else:
					assert False, 'Fail to setup outlet BC, possibly error in arterial network topology?.'
			if self.bcs[i].BC_TYPE != 'junction':
				print(f'info: i {i} type {self.bcs[i].BC_TYPE} side {self.bcs[i].BC_SIDE} vessel {self.bcs[i].V_ID} '
					f'bc_types {self.vessels[self.bcs[i].V_ID].bc_types[0]} {self.vessels[self.bcs[i].V_ID].bc_types[1]}')
			else:
				print(f'info: i {i} type {self.bcs[i].BC_TYPE} side {self.bcs[i].BC_SIDE} vessel {self.bcs[i].V_ID}')


	def step_inner_points(self):
		#for v in self.vessels:
		#	v.step_inner_points()
		self.pool.map(Vessel.step_inner_points, self.vessels)

	def step(self):
		self.step_inner_points()
		for i in range(len(self.bcs)):
			self.bcs[i].compute_bc(self)
		self.T += self.DT

	def run(self):
		steps = 0
		nsteps = int(self.T_FINAL / self.DT)
		cnt, T_cnt = 0, 1.0e-3
		T_last = self.T_FINAL - self.bcs[0].T_PER
		nsave = int(self.T_FINAL / T_cnt)+1
		Pmid = np.zeros(shape=(4,nsave))
		Pavg = np.zeros(shape=2)
		imid = np.array( [ int(self.vessels[i].N/2) for i in range(2) ])
		#imid[0] = (int) ( (1.0 - 0.5 / self.vessels[0].L) * self.vessels[0].N)
		#imid[2] = (int) (0.5 / self.vessels[2].L * self.vessels[2].N)
		while self.T < self.T_FINAL-1.0e-12:
			self.step()
			steps += 1
			if abs(self.T - (cnt+1)*T_cnt) < 0.1*self.DT:
				print(f'{self.T:.04g}/{self.T_FINAL}', end='\r')
				for i in range(2):
					Pmid[i][cnt] = self.vessels[i].pressure(self.vessels[i].S[imid[i]])
					Pmid[2+i][cnt] = self.vessels[i].U[imid[i]] * self.vessels[i].S[imid[i]]
					if self.T >= T_last:
						Pavg[i] += self.vessels[i].pressure(self.vessels[i].S[imid[i]]) * self.DT
				cnt += 1
		for i in range(2):
			Pavg[i] /= (self.T-T_last)
		x = np.linspace(0, self.T, nsave)
		print(f'Pavg: {Pavg} dP {Pavg[1]-Pavg[0]} FFR = Pavg1/Pavg0 = {Pavg[1]/Pavg[0]}')
  
		#id  = int(1.1 / 1e-4)
		# for i in range(len(self.vessels)):
		# 	print(f'vessel {i} times_inner_points: {self.vessels[i].tinn}')
  
		#PRESSURE
		plt.figure()
		plt.title('Pressure proximal')
		plt.plot(x[0:-2], Pmid[0][0:-2], label='vessel_1', color='b')
		#plt.plot(x, Pmid[1], label='Pmid_vessel_1', color='r')

		#np.savetxt('Pmid_0.txt', Pmid[0])
		#np.savetxt('Pmid_1.txt', Pmid[1]) 

		data = pd.read_csv('plot-data.csv')
		x = data['x'].to_numpy()
		y = data['y'].to_numpy()
		plt.plot(x, y, color='black', label='reference')
		plt.grid(visible=True)
		plt.xlabel('Time, s')
		plt.ylabel('P, mmHg')
		plt.legend()
		plt.savefig('pressure_test_3.png')
  
		#FLOW
		plt.figure()
		plt.title('Flow')
		plt.plot(x, Pmid[2], label='Pmid_vessel_0', color='b')
		plt.plot(x, Pmid[3], label='Pmid_vessel_1', color='r')

		# data_2 = pd.read_csv('flow.csv')
		# x_2 = data_2['x'].to_numpy()
		# y_2 = data_2['y'].to_numpy()  
  
		# plt.plot(x_2, y_2, color='black', label='reference')
		plt.grid(visible=True)
		plt.legend()
		plt.savefig('flow_test_2.png')

  
  		# plt.plot(x, Pmid[2], label='Pmid_vessel_0', color='b')
		# plt.plot(x, Pmid[3], label='Pmid_vessel_1', color='r')
		#plt.plot(x, Pmid[2+1], label='Pmid_vessel_1', color='g')

		#plt.plot(x, Pmid[3+2], label='Pmid_vessel_2', color='b')
		#plt.plot(x, Pmid[0]-Pmid[2], label='dPmid_vessel0-vessel2', color='k')
		#plt.xlim(T_last, self.T)



if __name__ == "__main__":
	if len(sys.argv) < 2:
		print(f'Usage: python3 {sys.argv[0]} run/test.json')
		exit()
	model = FlowModel(sys.argv[1])
	model.run()

import numpy as np  # numpy를 np로 축약하여 사용
import numpy  # numpy 모듈 전체를 사용
import pandas as pd  # pandas를 pd로 축약하여 사용
import pandas  # pandas 모듈을 전체 이름으로 사용
import datetime  # datetime 모듈 전체를 사용
from dateutil.relativedelta import relativedelta  # relativedelta만 사용
import functools  # functools 모듈 전체를 사용
import time  # time 모듈 전체를 사용
import copy  # copy 모듈 전체를 사용
import psutil  # psutil 모듈 전체를 사용
from numba import jit  # numba에서 jit 함수만 사용
from multiprocessing import Pool  # multiprocessing 모듈에서 Pool만 사용
import matplotlib.pyplot as plt  # matplotlib.pyplot을 plt로 축약하여 사용
from scipy.stats import norm  # scipy.stats에서 norm만 사용
import matplotlib.ticker as mtick  # matplotlib.ticker을 mtick으로 축약하여 사용
import scipy.stats as stats  # scipy.stats 모듈을 stats로 축약하여 사용
import calendar
import numba
import os
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
import multiprocessing
import sys
#########################################################################################################

class FuturesDataFetcher:
# /content/drive/MyDrive/대학원/졸업논문/SOFR/SONIA/1-month-sonia-prices-end-of-day-01-02-2020.csv
    def __init__(self, date, directory=r'C:/Users/PC2212/Desktop/논문/SOFR/Code/SOFR/'):
        self.dir = directory
        self.fn = '-month-sofr-prices-end-of-day-%s.csv' % date.strftime('%m-%d-%Y')
        self.fixings_fn = r'%s\sofr_fixings.csv' % self.dir
        self.sofr1m_data = self.get_sofr_data_table(1)
        self.sofr3m_data = self.get_sofr_data_table(3)
        self.fixings = self.get_fixings()
    def get_sofr_data_table(self, tenor):
        return pandas.read_csv('%s%s%s' % (self.dir, tenor, self.fn), engine='python')

    def get_fixings(self):
        fixings_df = pandas.read_csv(self.fixings_fn, parse_dates=['Date'])
        fixings_df['Date'] = fixings_df['Date'].dt.date
        dt = min(fixings_df.Date)
        last_date = max(fixings_df.Date)
        fixings_dict = {r['Date']: r['Rate'] for i, r in fixings_df.iterrows()} # 영업일 SOFR
        fixings_dict_filled = {}
        sofr = 0.0
        while dt <= last_date:
            sofr = fixings_dict.get(dt, sofr)
            fixings_dict_filled[dt] = sofr
            dt += relativedelta(days=1)
        return fixings_dict_filled # 모든 날짜 SOFR
    
class Future:
    def __init__(self, tenor, code, market_price, market_date, fixings_dict):
        self.tenor = tenor
        self.code = code
        self.market_price = market_price
        self.market_date = market_date
        # 연도 추출 (code의 마지막 두 자리가 연도를 나타냄)
        self.year = 2000 + int(self.code[-2:])
        # 월 추출 (code에서 연도 바로 앞의 알파벳이 월을 나타냄)
        self.month = self.get_month_from_code(self.code[-3:-2])
        self.reference_start = self.get_reference_date('start')
        self.reference_end = self.get_reference_date('end')
        self.min_tick = self.get_min_tick()
        self.reference_days = (self.reference_end - self.reference_start).days + 1
        self.all_dates = [self.reference_start + relativedelta(days=d) for d in range(self.reference_days)] # ref_start부터 end까지의 모든 날짜
        self.n_days = len(self.all_dates)
        self.accrual_days = [dt for dt in self.all_dates if dt < market_date] # #all_dates에서 dt라는 이름으로 하나씩 날짜를 가져와서 dt < market_date인지 확인한 후 조건을 만족하는 dt만 accrual_days 리스트에 추가
        self.fixings = [fixings_dict[dt] for dt in self.accrual_days]
        self.accrued = self.get_accrued()
        self.forward_dates = [dt for dt in self.all_dates if dt not in self.accrual_days] # st_date부터 reference_end까지 날짜들
        self.model_price = None
        self.model_forwards = None
        self.f_t = self.get_f_t()

    def get_month_from_code(self, code_letter):
        """ 알파벳으로부터 월을 반환하는 함수 """
        month_mapping = {
            'F': 1,  # January
            'G': 2,  # February
            'H': 3,  # March
            'J': 4,  # April
            'K': 5,  # May
            'M': 6,  # June
            'N': 7,  # July
            'Q': 8,  # August
            'U': 9,  # September
            'V': 10, # October
            'X': 11, # November
            'Z': 12  # December
        }
        return month_mapping[code_letter]

    def get_accrued(self):
        # st_date까지의 SOFR금리를 1+f/360 한 후 전부 곱함, 식(60)
        if self.tenor == 'Q':
            return numpy.prod([1 + f / 360 for f in self.fixings])
        # 식(55)
        elif self.tenor == 'M':
            return sum(self.fixings)

    def get_min_tick(self):
        if self.tenor == 'Q':
            return 0.005 if self.reference_end + relativedelta(months=-4) > self.market_date else 0.0025
        elif self.tenor == 'M':
            return 0.005 if self.reference_start > self.market_date else 0.0025

    def get_reference_date(self, date_type):
        if self.tenor == 'Q':
            dt = datetime.date(self.year, self.month, 15)
            # market_date이 속한 월의 15일 이후로 가장 가까운 수요일
            if date_type == 'start':
                return dt.replace(day=15 + (2 - dt.weekday()) % 7)
            # market_date이 속한 월에서 3개월 후, 가장 가까운 수요일의 전날인 화요일
            elif date_type == 'end':
                end_dt = dt + relativedelta(months=3)
                return end_dt.replace(day=15 + (2 - end_dt.weekday()) % 7) + relativedelta(days=-1)
        elif self.tenor == 'M':
            if date_type == 'start':
                return datetime.date(self.year, self.month, 1)
            elif date_type == 'end':
                return datetime.date(self.year, self.month, calendar.monthrange(self.year, self.month)[1])

    def price_from_dict(self, forward_curve_dict):
        # forward_curve_dict : -1이면 fixings값을 쓰고 그렇기 않으면 levels값을 써라
        return self.price([forward_curve_dict[dt] for dt in self.all_dates]) # price_mc를 쓰고자하면 price->price_mc, 단 self.model_price = 100필요

    def price_mc(self, forwards):
        return self.price(forwards) - 100 + self.model_price

    def price(self, forwards):
        # 식(59)
        return self.price_q(forwards) if self.tenor == 'Q' else 100 - 100 * (self.accrued + sum(forwards)) \
                                                                / self.n_days

    def price_q(self, forwards):
        # 60, 61
        return 100-(numpy.prod([1 + f / 360 for f in forwards]) * self.accrued - 1) * 36000 / self.n_days

    def raw_price(self, forwards):
        # 61, 62
        return (numpy.prod([1 + f / 360 for f in forwards]) - 1) * 36000 / self.n_days

    def get_f_t(self):
        # f_t의 길이는 st_date부터 st_date이 속한 월의 마지막일까지의 날짜수, Max=31개
        return [(dt-self.market_date).days/365.25 for dt in self.forward_dates]
    
class ForwardCurve:
    def __init__(self, date, start_date, last_date, directory=r'C:/Users/PC2212/Desktop/논문/SOFR/Code/SOFR/'):
        self.directory = directory
        self.meeting_date_fn = r'%s\fomc_dates.csv' % self.directory
        self.value_date = date
        self.start_date = start_date
        self.last_date = last_date
        self.meeting_dates = self.get_meeting_dates()
        self.levels = [0.0] * (len(self.meeting_dates) + 1)
        self.fixings = FuturesDataFetcher(date, self.directory).fixings
        self.period = [dt.date() for dt in pandas.date_range(start_date, last_date, freq='d')]
        self.period_dictionary = self.get_period_dictionary()
        self.forward_curve_dict = self.get_forward_curve_dict()


    def get_meeting_dates(self):
        meeting_dates = pandas.read_csv(self.meeting_date_fn, parse_dates=['Date'])['Date'].dt.date.tolist()
        #md는 val_date과 last_date 사이에 FOMC 회의날짜
        filtered_dates = [md for md in meeting_dates if self.last_date > md >= self.value_date] #val_date을 첫번쨰 FOMC회의로 보는 문제를 수정하기 위해 수정한코드
        return filtered_dates


    def get_meeting_dates_yf(self, dt=None):
        # 주어진 날짜(또는 기본적으로 self.value_date)를 기준으로 FOMC 회의 날짜들과의 기간을 연 단위로 계산하여 반환
        if dt:
            meeting_dates = pandas.read_csv(self.meeting_date_fn, parse_dates=['Date'])['Date'].dt.date.tolist()
            md = [md for md in meeting_dates if self.last_date > md >= dt]
            meeting_dates_yf = [(d - dt).days / 365.0 for d in md]

            return meeting_dates_yf
        else:
            meeting_dates_yf = [(d - self.value_date).days / 365.0 for d in self.meeting_dates]

            return meeting_dates_yf


    def get_period_dictionary(self):
        # meeting_date_array = [[val_date - 1일], [val_date과 end_date(=last_date) 사이의 FOMC 날짜]]
        meeting_date_array = numpy.array([self.value_date + relativedelta(days=-1)] + self.meeting_dates) #기존코드
        # st_date부터 end_date까지의 날짜들이 담긴 self.period를 키로하고, 해당 날짜가 몇번째 FOMC 회의 이후에 있는지를 나타내는 값을 가짐
        # {st_date : -1}, {st_date 다음날~FOMC회의 전날 : 0}, {FOMC회의날~end_date : 1}
        return {dt: len(meeting_date_array[meeting_date_array < dt]) - 1 for dt in self.period}

    def get_forward_curve_dict(self):
        curve_dict = {}
        for dt, idx in self.period_dictionary.items():
        # -1이면 fixings값을 쓰고 그렇기 않으면 levels값을 써라
            curve_dict[dt] = self.fixings[dt] if idx == -1 else self.levels[idx]
        return curve_dict

    def calibrate(self, futures_set):
        # -1이면 fixings값을 쓰고 0이면 levels[0]을 쓰고 1이면 levels[1]을 써라
        self.levels = pandas.read_csv('%s%s_levels.csv' %
                                      (self.directory, self.value_date.strftime('%Y%m%d')))['Rate'].tolist()
        self.forward_curve_dict = self.get_forward_curve_dict()
        for fut in futures_set.futures_set:
            fut.model_price = fut.price_from_dict(self.forward_curve_dict)
            fut.model_forwards = numpy.array([self.forward_curve_dict[dt] for dt in fut.forward_dates])


class FuturesSet:
    def __init__(self, date, last_date, directory=r'C:/Users/PC2212/Desktop/논문/SOFR/Code/SOFR/'):
        self.directory = directory
        self.dt = date
        self.last_date = last_date
        self.futures_set = self.get_futures()
        # self.futures_price_dict = {f.code[:6].lower(): f.market_price for f in self.futures_set}
        # self.futures_dict = {f.code[:6].lower(): f for f in self.futures_set}
        self.futures_price_dict = {f.code[:6]: f.market_price for f in self.futures_set}
        self.futures_dict = {f.code[:6]: f for f in self.futures_set}

    def get_futures(self):
        futures_df = FuturesDataFetcher(self.dt, self.directory)
        all_futures = [Future('M', r['Contract'], r['Last'], self.dt, futures_df.fixings)
                      for idx, r in futures_df.sofr1m_data.iterrows() if r['Contract'].startswith('SER')] + \
                      [Future('Q', r['Contract'], r['Last'], self.dt, futures_df.fixings)
                      for idx, r in futures_df.sofr3m_data.iterrows() if r['Contract'].startswith('SFR')]
        
        return [fut for fut in all_futures if fut.reference_end <= self.last_date]


    def get_dataframe(self):
        futures_df = pandas.DataFrame(columns=['start_date', 'end_date'], index=[fut.code for fut in self.futures_set])
        for fut in self.futures_set:
            futures_df.loc[fut.code] = pandas.Series({'start_date': fut.reference_start, 'end_date': fut.reference_end})
        return futures_df
    
###########################################################

class StochasticIntegral:
    def __init__(self, path_generator, paths=100):
        self.paths = paths
        self.path_generator = path_generator

    def calculate_stochastic_integral(self, t, f_t, future=None, raw=False):
        if future:
            f_t = future.f_t
            if not raw:
                # calculate path terminal values as futures prices for all paths
                path_terminal_values = \
                    numpy.array([future.price(self.path_generator.path_terminal_value(t, f_t) +
                                              future.model_forwards) for _ in range(self.paths)]).T
                path_terminal_values -= numpy.average(path_terminal_values, axis=0) - future.market_price
            else:
                t = f_t[-1]
                path_terminal_values = numpy.array([future.raw_price(self.path_generator.path_terminal_value(t, f_t))
                                                    for _ in range(self.paths)]).T
                return numpy.std(path_terminal_values, axis=0) / numpy.sqrt(t)
        else:
            path_terminal_values = numpy.array([self.path_generator.path_terminal_value(t, f_t)
                                                for _ in range(self.paths)])
        return {'mean': numpy.average(path_terminal_values, axis=0), 'std': numpy.std(path_terminal_values, axis=0),
                'terminal_values': path_terminal_values}


class Path:
    def __init__(self, sigma_function, correlation, alpha, theta, n_steps=10, drift=False):
        self.sigma_function = sigma_function
        self.nSteps = n_steps
        self.correlation = correlation
        self.alpha = alpha
        self.theta = theta
        self.drift = drift
        self.sv_path = []

    @functools.cache
    def steps(self, t):
        step_size = t / self.nSteps
        return [i * step_size for i in range(1, self.nSteps + 1)]

    @functools.cache
    def dts(self, t):
        return numpy.diff([0] + self.steps(t))

    @functools.cache
    def sigma_paths(self, t, f_t):
        return [sig.sigma(self.steps(t), f_t) for sig in self.sigma_function]

    @functools.cache
    def sigma_paths_mult_f_t(self, t, f_t):
        # f_t는 각 경로의 시간 변화
        sigma_paths = numpy.array([self.sigma_paths(t, f_t_) for f_t_ in f_t])
        return sigma_paths

        #return numpy.array([self.sigma_paths(t, f_t_) for f_t_ in f_t])

    @functools.cache
    def rs_corr(self, corr):
        return numpy.sqrt(1 - corr ** 2)

    def path_terminal_value(self, t, f_t):
        # set T collection
        f_t = f_t if isinstance(f_t, list) else [f_t]
        # calculate sigma paths for T collection
        sigma_paths = self.sigma_paths_mult_f_t(t, tuple(f_t))
        #print(f"sigma_paths at t={t}: {sigma_paths}")
        # calculate stochastic component
        d_paths = numpy.array([self.d_path(t, corr, alpha, theta) for corr, alpha, theta in
                               zip(self.correlation, self.alpha, self.theta)])
        #print(f"d_paths at t={t}: {d_paths}")
        # multiply stochastic component with deterministic sigma paths
        # aggregate factor and aggregate stochastic integral
        terminal_value = numpy.sum(numpy.sum(d_paths * sigma_paths, axis=1), axis=1)
        #print(f"terminal_value at t={t}: {terminal_value}")
        # add drift
        if self.drift:
            drifts = self.get_drift(t, tuple(f_t))
            terminal_value += drifts
        return terminal_value

    @functools.cache
    def get_drift(self, t, f_t):
        dts = self.dts(t)
        # get drift for each T
        drifts = []
        for f_t_ in f_t:
            drift = 0.0
            # sum across factors
            for sv, sig in zip(self.sv_path, self.sigma_function):
                # outer integral
                outer_sigmas = sig.sigma(self.steps(t), f_t_)
                inner_integrals = []
                # inner integral
                for t_ in self.steps(t):
                    step_size = (f_t_ - t_) / 10.0
                    steps_ = [t_ + i * step_size for i in range(1, 11)]
                    inner_integrals.append(sum([sig.sigma([t_], ft)[0] * step_size for ft in steps_]))
                drift += numpy.sum(numpy.array(inner_integrals) * outer_sigmas * dts)
            drifts.append(drift)
        return drifts

    def d_path(self, t, corr, alpha, theta):
        draw = get_two_factor_draw(corr, len(self.steps(t)), self.rs_corr(corr))
        sv_path = stochastic_vol_path(draw[1], self.dts(t), numba.typed.List(alpha[0]),
                                      numba.typed.List(alpha[1]), theta)
        self.sv_path.append(sv_path)
        return draw[0] * sv_path


@jit(nopython=True)
def get_two_factor_draw(corr, size, src):
    x1 = numpy.random.normal(0, 1, size)
    return x1, corr * x1 + src * numpy.random.normal(0, 1, size)


@jit(nopython=True)
def stochastic_vol_path(draw, dts, alpha_list, alpha_threshold_list, theta):
    v = [1]
    alpha_idx = 0
    running_t = 0.0
    for dt, du in zip(dts, draw):
        if running_t > alpha_threshold_list[alpha_idx]:
            alpha_idx += 1
        v.append(v[-1] + theta * (1 - v[-1]) * dt + alpha_list[alpha_idx] * numpy.sqrt(numpy.abs(v[-1]) * dt) * du)
        running_t += dt
    v_result = [1]
    for dt, v_ in zip(dts, v[1:]):
        v_result.append(numpy.sqrt(dt) * v_)
    return numpy.array(v_result[1:])


class Sigma:
    def __init__(self, sigma_0, chi=None, phi=None, indicator=None, factor=0):
        self.sigma_0 = sigma_0
        self.chi_function = chi
        self.phi_function = phi
        self.indicator_function = indicator
        self.factor = factor

    def sigma(self, steps, f_t):
        return self.chi_function.chi(steps) * self.phi_function.phi(f_t) * \
               self.indicator_function.indicator(steps, f_t, self.factor)


class Phi:
    def __init__(self, lambda_=0.0):
        self.lambda_function = lambda_
        if isinstance(lambda_, float):
            self.phi = self.constant_lambda_phi

    @functools.cache
    def constant_lambda_phi(self, f_t):
        return numpy.exp(-self.lambda_function * f_t)


class Chi:
    def __init__(self, lambda_=0.0):
        self.lambda_function = lambda_
        if isinstance(lambda_, float):
            self.chi = self.constant_lambda_chi

    def constant_lambda_chi(self, steps):
        return numpy.array([numpy.exp(self.lambda_function * t) for t in steps])


class Indicator:
    def __init__(self, m_dates, gamma_, factor_vols):
        self.meeting_dates = m_dates
        self.gamma = gamma_
        self.factor_vols = factor_vols

    def indicator(self, steps, f_t, factor):
        #print(f"f_t: {f_t}, type: {type(f_t)}")
        a_s = [len(list(filter(lambda x: f_t > x >= t, self.meeting_dates))) for t in steps]

        indicators = []
        for a in a_s:
            indicators.append(0)
            for i in range(a):
                indicators[-1] = indicators[-1] + self.gamma[factor, i] * self.factor_vols[factor]

        return indicators


def meeting_dates_from_file(start_date, file_name):
    return [(d - start_date).days / 365.0 for d in
            pandas.read_csv(file_name, parse_dates=['Date'])['Date'].dt.date.tolist()]


def gamma():
    gamma_values = numpy.loadtxt(r'C:\Users\PC2212\Desktop\v_jump.csv', delimiter=',')

    return gamma_values


def empirical_vol():
    return numpy.sqrt(numpy.loadtxt(r'C:\Users\PC2212\Desktop\s_jump.csv', delimiter=',') * 365)


class Runner:
    def __init__(self, **kwargs):
        self.t = None
        self.paths = None
        self.theta = None
        self.alpha = None
        self.corr = None
        self.steps = None
        self.m_dates = None
        self.factors = None
        self.lambdas = None
        self.f_t = None
        self.future = None
        self.drift = None
        self.raw_future = False
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not hasattr(self, 'f_t_0'):
            self.f_t_0 = self.f_t
        if not hasattr(self, 'factor_vols'):
            self.factor_vols = empirical_vol()

    def get_results(self):
        self.corr = self.corr[:self.factors]
        self.theta = self.theta[:self.factors]
        self.alpha = self.alpha[:self.factors]
        self.lambdas = self.lambdas[:self.factors]
        #process alpha
        self.alpha = [(a[0], a[1] + [999.9]) if isinstance(a, tuple) else ([a], [999.9]) for a in self.alpha]
        # phi and chi functions
        phi = [Phi(self.lambdas[i]) for i in range(self.factors)]
        chi = [Chi(self.lambdas[i]) for i in range(self.factors)]
        # get eigenvectors (gamma)
        g = gamma()
        # indicator function
        ind = Indicator(self.m_dates, g, self.factor_vols)
        # sigma functions
        sigma_f = [Sigma(1.0, chi=chi[i], phi=phi[i], indicator=ind, factor=i) for i in range(self.factors)]
        self.f_t_0 = self.f_t
        # path generator
        p = Path(sigma_f, n_steps=self.steps, correlation=self.corr,
                 alpha=self.alpha, theta=self.theta, drift=self.drift)
        # stochastic integral generate #NOTE this includes initial term structure and drift
        w = StochasticIntegral(p, paths=self.paths)
        return w.calculate_stochastic_integral(self.t, self.f_t, self.future, self.raw_future)

    def get_option_results(self, strikes_types):
        results = self.get_results()
        return {kt: price_option(kt, results['terminal_values']) for kt in strikes_types}


def price_option(kt, mc):
    pc = 1 if kt[1] == 'C' else -1
    payoff = numpy.maximum((mc - kt[0]) * pc, 0.0)
    average = numpy.average(payoff)
    variance = numpy.sum([(p - average) ** 2 for p in payoff]) / (len(payoff) - 1)
    interval_width = 1.96 * numpy.sqrt(variance / len(payoff))
    return average, (average - interval_width, average + interval_width)

###############################################################################################




def calibration(args, scan_variable, scan_start, scan_inc, parallel=True):
    arg_list = []
    n = 2
    variables = [scan_start + i * scan_inc for i in range(n)]
    for v in variables:
        tmp_args = copy.deepcopy(args)
        if len(scan_variable) == 2:
            tmp_args[scan_variable[0]][scan_variable[1]] = v
        elif len(scan_variable) == 3:
            #print(f"Setting {scan_variable[0]}[{scan_variable[1]}][0][{scan_variable[2]}] = {v}")
            tmp_args[scan_variable[0]][scan_variable[1]][0][scan_variable[2]] = v
        arg_list.append(tmp_args)

    if parallel:
        with Pool(max(len(arg_list), n)) as p:
            return p.map(parallel_runner_wrap, arg_list), variables
    else:
        return [Runner(**a).get_option_results(a['strike_types']) for a in arg_list], variables


def calibration_2d(args, variables, parallel=True):
    arg_list = []
    n = 2
    var_1 = variables[0]
    var_2 = variables[1]
    variables_1 = [var_1[1] + i * var_1[2] for i in range(n)]
    variables_2 = [var_2[1] + i * var_2[2] for i in range(n)]
    outvar1 = []
    outvar2 = []
    for v1 in variables_1:
        for v2 in variables_2:
            tmp_args = copy.deepcopy(args)
            if len(var_1[0]) == 2:
                tmp_args[var_1[0][0]][var_1[0][1]] = v1
            elif len(var_1[0]) == 3:
                tmp_args[var_1[0][0]][var_1[0][1]][0][var_1[0][2]] = v1
            if len(var_2[0]) == 2:
                tmp_args[var_2[0][0]][var_2[0][1]] = v2
            elif len(var_2[0]) == 3:
                tmp_args[var_2[0][0]][var_2[0][1]][0][var_2[0][2]] = v2
            arg_list.append(tmp_args)
            outvar1.append(v1)
            outvar2.append(v2)
    if parallel:
        with Pool(max(len(arg_list), n)) as p:
            return p.map(parallel_runner_wrap, arg_list), outvar1, outvar2
    else:
        return [Runner(**a).get_option_results(a['strike_types']) for a in arg_list], outvar1, outvar2


def calibration_nodes(args, test_variables, variable_idx, nodes, parallel=True):
    arg_list = []
    n = 2
    for v in nodes:
        tmp_args = copy.deepcopy(args)
        tmp_args[test_variables[0]][variable_idx[0]] = v[0]
        tmp_args[test_variables[1]][variable_idx[1]] = v[1]
        arg_list.append(tmp_args)
    if parallel:
        with Pool(max(len(arg_list), n)) as p:
            return p.map(parallel_runner_wrap, arg_list)
    else:
        return [Runner(**a).get_option_results(a['strike_types']) for a in arg_list]


def calibration_2(args, test_variables, variable_idx, var1_range, var2_range, parallel=True):
    arg_list = []
    n = 2
    xx = 6
    var_space_1 = (var1_range[1] - var1_range[0]) / (xx-1)
    var_space_2 = (var2_range[1] - var2_range[0]) / (xx-1)
    variable_list_1 = [var1_range[0] + var_space_1 * i for i in range(xx)]
    variable_list_2 = [var2_range[0] + var_space_2 * i for i in range(xx)]
    variables = []
    for v1 in variable_list_1:
        for v2 in variable_list_2:
            tmp_args = copy.deepcopy(args)
            tmp_args[test_variables[0]][variable_idx[0]] = v1
            tmp_args[test_variables[1]][variable_idx[1]] = v2
            arg_list.append(tmp_args)
            variables.append((v1, v2))
    if parallel:
        with Pool(max(len(arg_list), n)) as p:
            return p.map(parallel_runner_wrap, arg_list), variables
    else:
        return [Runner(**a).get_option_results(a['strike_types']) for a in arg_list], variables


def parallel_runner_wrap(args):
    return Runner(**args).get_option_results(args['strike_types'])


def convert_option_data(date, futures_prices):
    source_dir = r'C:/Users/PC2212/Desktop/논문/SOFR/Code/SOFR/raw_data/%s/' % date.strftime('%Y%m%d')
    target_dir = r'C:/Users/PC2212/Desktop/논문/SOFR/Code/SOFR/results/%s/' % date.strftime('%Y%m%d')
    options_3m = []
    for fn in os.listdir(source_dir):    # source_dir 디렉토리 내의 모든 파일을 읽어옵니다
        if fn.startswith('SER') or fn.startswith('SFR'):    # 파일명이 'SER' 또는 'SFR'로 시작하는 경우에만 데이터를 처리
            # !!! skipfooter=1 옵션 지워버림 !!!
            raw_data_df = pandas.read_csv('%s%s' % (source_dir, fn), engine='python')[['Strike', 'Last']] # skipfooter=1 옵션을 사용하여 마지막 행을 무시
            option_code = fn.split('_')[0]    # 'SERQ4C' 또는 'SFRQ4C' 형태의 옵션 코드 추출
            expiry = fn.split('_')[1].split('.')[0]  # '20240903' 형태의 만기일 추출

            raw_data_df['Type'] = raw_data_df.apply(lambda row: row['Strike'][-1], axis=1)
            raw_data_df['Strike'] = raw_data_df.apply(lambda row: float(row['Strike'][:-1]), axis=1)
            #print("futures_price : ", futures_prices)
            atm_rate = futures_prices[option_code]
            target_df = raw_data_df[((raw_data_df['Type'] == 'C') & (raw_data_df['Strike'] >= atm_rate)) |
                                    ((raw_data_df['Type'] == 'P') & (raw_data_df['Strike'] <= atm_rate))]
            max_strike_p_seq = target_df.query("`Last` == 0.0025 and `Type` == 'P'")['Strike']
            max_strike_p = 0 if len(max_strike_p_seq) == 0 else max(max_strike_p_seq)
            min_strike_c_seq = target_df.query("`Last` == 0.0025 and `Type` == 'C'")['Strike']
            min_strike_c = 200.0 if len(min_strike_c_seq) == 0 else min(min_strike_c_seq)
            target_df = target_df[((target_df['Type'] == 'C') & (target_df['Strike'] <= min_strike_c)) |
                                  ((target_df['Type'] == 'P') & (target_df['Strike'] >= max_strike_p))]
            target_fn = '%s%s_%s.csv' % (target_dir, expiry, option_code)
            target_df.to_csv(target_fn, index=False)
            if fn.startswith('SFR'):    # 3M 옵션이면 options_3m 리스트에 추가
                options_3m.append((option_code, expiry, target_fn))
    return options_3m, target_dir


def calibrate_system_2(params, test_variables, variable_idx, f_vol_range, lambda_range,
                       expiry_months=None, underlying_list=None):
    st_date = datetime.date(2023, 3, 15)
    e_date = datetime.date(2024, 9, 17)  
    val_date = datetime.date(2023, 6, 16)
    f = FuturesSet(params['val_date'], e_date)
    fc = ForwardCurve(params['val_date'], st_date, e_date)
    fc.calibrate(f)
    options, target_dir_ = convert_option_data(params['val_date'], f.futures_price_dict)
    options_system_df = None
    confidence_interval_dict = {}
    variables_ = None
    for o in options:
        if not underlying_list or o[0] in underlying_list:
            exp_date = datetime.datetime.strptime(o[1], '%Y%m%d').date()
            if not expiry_months or exp_date.month in expiry_months:
                tmp_params = copy.deepcopy(params)
                print((o[0], o[1]))
                underlying = o[0]
                tmp_params['t'] = (exp_date - val_date).days / 365.25
                tmp_params['future'] = f.futures_dict[o[0]]
                fn_ = r'C:/Users/PC2212/Desktop/논문/SOFR/Code/SOFR/results/%s/%s_%s.csv' % \
                      (val_date.strftime('%Y%m%d'), o[1], underlying)
                option_df = pandas.read_csv(fn_)[['Strike', 'Type', 'Last']]
                all_st = [(k, pc, mp) for k, pc, mp in zip(option_df['Strike'], option_df['Type'], option_df['Last'])]
                st = [(0.0, 'P'), (200.0, 'C')]
                for st_ in all_st:
                    st[0] = st_ if st_[1] == 'P' and st_[0] > st[0][0] else st[0]
                    st[1] = st_ if st_[1] == 'C' and st_[0] < st[1][0] else st[1]
                tmp_params['strike_types'] = st
                prices = []
                res, variables_ = calibration_2(tmp_params, test_variables, variable_idx, f_vol_range, lambda_range)
                for st_ in st:
                    prices.append([r[st_][0] for r in res])
                    for r, v in zip(res, variables_):
                        confidence_interval_dict[(o[0], exp_date, st_[0], st_[1], v)] = r[st_][1]
                if not isinstance(options_system_df, pandas.DataFrame):
                    cols = ['Expiry', 'Underlying', 'Type', 'Strike', 'Price'] + variables_
                    options_system_df = pandas.DataFrame(columns=cols)
                options_system_df.loc[len(options_system_df)] = \
                    [exp_date, underlying, st[0][1], st[0][0], st[0][2]] + prices[0]
                options_system_df.loc[len(options_system_df)] = \
                    [exp_date, underlying, st[1][1], st[1][0], st[1][2]] + prices[1]
    options_system_df.to_csv('%s%s.csv' % (target_dir_, 'options_system'), index=False)
    best_variable = variables_[0]
    best_error = 1e6
    for v in variables_:
        total_error = sum((abs(options_system_df[v] - options_system_df['Price']) - 0.005).clip(lower=0.0))
        if total_error < best_error:
            best_error = total_error
            best_variable = v
    print("best_variable")
    print(best_variable)
    print("best error :", best_error)
    best_result_df = copy.deepcopy(
        options_system_df[['Expiry', 'Underlying', 'Type', 'Strike', 'Price', best_variable]])
    best_result_df['Bid'] = best_result_df['Price'] - 0.0025
    best_result_df['Offer'] = best_result_df['Price'] + 0.0025
    best_result_df['Lower CI'] = best_result_df.apply(lambda row: confidence_interval_dict[
        (row['Underlying'], row['Expiry'], row['Strike'], row['Type'], best_variable)][0], axis=1)
    best_result_df['Upper CI'] = best_result_df.apply(lambda row: confidence_interval_dict[
        (row['Underlying'], row['Expiry'], row['Strike'], row['Type'], best_variable)][1], axis=1)
    best_result_df['Bid Vol'] = best_result_df.apply(lambda row: implied_vol_from_df_row(row, params, 'Bid', f), axis=1)
    best_result_df['Offer Vol'] = best_result_df.apply(lambda row: implied_vol_from_df_row(row, params, 'Offer', f),
                                                       axis=1)
    best_result_df['Lower CI Vol'] = best_result_df.apply(lambda row:
                                                          implied_vol_from_df_row(row, params, 'Lower CI', f), axis=1)
    best_result_df['Upper CI Vol'] = best_result_df.apply(lambda row:
                                                          implied_vol_from_df_row(row, params, 'Upper CI', f), axis=1)
    best_result_df.to_csv('%s%s.csv' % (target_dir_, 'best_result'), index=False)
    implied_vol_df = best_result_df.groupby(['Expiry', 'Underlying']). \
        agg({'Expiry': 'first', 'Bid Vol': 'mean', 'Offer Vol': 'mean', 'Lower CI Vol': 'mean', 'Upper CI Vol': 'mean'})
    implied_vol_df.to_csv('%s%s.csv' % (target_dir_, 'implied_vol_df'), index=False)
    plt.fill_between(implied_vol_df['Expiry'], implied_vol_df['Bid Vol'], implied_vol_df['Offer Vol'],
                     color='black', label='market bid/offer')
    plt.fill_between(implied_vol_df['Expiry'], implied_vol_df['Lower CI Vol'],
                     implied_vol_df['Upper CI Vol'], color='red', alpha=0.2, label='model ci=95%')
    plt.ylabel('implied normal vol (bp)')
    plt.xlabel('expiry')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()
    # raw_data_df['Type'] = raw_data_df.apply(lambda row: row['Strike'][-1], axis=1)


def calibrate_system_bisection(params, test_variables, variable_idx, var_1_range, var_2_range, expiry_months=None,
                               underlying_list=None):
    x1, x2 = var_1_range
    y1, y2 = var_2_range
    current_nodes = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

    errors = get_errors(params, test_variables, variable_idx, current_nodes, expiry_months, underlying_list)

    for i in range(10):
        print("i=", i)
        print("old_current_nodes :", current_nodes)
        print("old errors :", errors)
        min_idx = errors.index(min(errors))
        if min_idx == 0:
            x2 = 0.5 * (x1 + x2)
            y2 = 0.5 * (y1 + y2)
        elif min_idx == 1:
            x1 = 0.5 * (x1 + x2)
            y2 = 0.5 * (y1 + y2)
        elif min_idx == 2:
            x2 = 0.5 * (x1 + x2)
            y1 = 0.5 * (y1 + y2)
        elif min_idx == 3:
            x1 = 0.5 * (x1 + x2)
            y1 = 0.5 * (y1 + y2)
        current_nodes = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        errors = get_errors(params, test_variables, variable_idx, current_nodes, expiry_months, underlying_list)

    print("Final_current_nodes :", current_nodes)
    print("Final errors :", errors)


def get_errors(params, test_variables, variable_idx, nodes, expiry_months, underlying_list):
    st_date = datetime.date(2023, 3, 15)
    e_date = datetime.date(2024, 9, 17)  
    val_date = datetime.date(2023, 6, 16)
    f = FuturesSet(params['val_date'], e_date)
    fc = ForwardCurve(params['val_date'], st_date, e_date)
    fc.calibrate(f)
    options, target_dir_ = convert_option_data(params['val_date'], f.futures_price_dict)
    options_system_df = None
    confidence_interval_dict = {}
    for o in options:
        if not underlying_list or o[0] in underlying_list:
            exp_date = datetime.datetime.strptime(o[1], '%Y%m%d').date()
            if not expiry_months or exp_date.month in expiry_months:
                tmp_params = copy.deepcopy(params)
                #print((o[0], o[1]))
                underlying = o[0]
                tmp_params['t'] = (exp_date - val_date).days / 365.25
                tmp_params['future'] = f.futures_dict[o[0]]
                fn_ = r'C:/Users/PC2212/Desktop/논문/SOFR/Code/SOFR/results/%s/%s_%s.csv' % \
                      (val_date.strftime('%Y%m%d'), o[1], underlying)
                option_df = pandas.read_csv(fn_)[['Strike', 'Type', 'Last']]
                all_st = [(k, pc, mp) for k, pc, mp in zip(option_df['Strike'], option_df['Type'], option_df['Last'])]
                st = [(0.0, 'P'), (200.0, 'C')]
                for st_ in all_st:
                    st[0] = st_ if st_[1] == 'P' and st_[0] > st[0][0] else st[0]
                    st[1] = st_ if st_[1] == 'C' and st_[0] < st[1][0] else st[1]
                tmp_params['strike_types'] = st
                prices = []
                res = calibration_nodes(tmp_params, test_variables, variable_idx, nodes)
                for st_ in st:
                    prices.append([r[st_][0] for r in res])
                    for r, v in zip(res, nodes):
                        confidence_interval_dict[(o[0], exp_date, st_[0], st_[1], v)] = r[st_][1]
                if not isinstance(options_system_df, pandas.DataFrame):
                    cols = ['Expiry', 'Underlying', 'Type', 'Strike', 'Price'] + nodes
                    options_system_df = pandas.DataFrame(columns=cols)
                options_system_df.loc[len(options_system_df)] = \
                    [exp_date, underlying, st[0][1], st[0][0], st[0][2]] + prices[0]
                options_system_df.loc[len(options_system_df)] = \
                    [exp_date, underlying, st[1][1], st[1][0], st[1][2]] + prices[1]
    errors = [sum((abs(options_system_df[v] - options_system_df['Price']) - 0.005).clip(lower=0.0)) for v in nodes]
    return errors


def calibrate_system(params, test_variable, start_variable, increment, underlying_list=None, expiry_months=None):
    val_date = datetime.date(2023, 6, 16)
    st_date = datetime.date(2023, 3, 15)
    e_date = datetime.date(2024, 9, 17)
    f = FuturesSet(params['val_date'], e_date)
    fc = ForwardCurve(params['val_date'], st_date, e_date)
    fc.calibrate(f)
    options, target_dir_ = convert_option_data(params['val_date'], f.futures_price_dict)
    options_system_df = None
    confidence_interval_dict = {}
    variables_ = None
    for o in options:
        if not underlying_list or o[0] in underlying_list:
            exp_date = datetime.datetime.strptime(o[1], '%Y%m%d').date()
            if not expiry_months or exp_date.month in expiry_months:
                tmp_params = copy.deepcopy(params)
                print((o[0], o[1]))
                exp_date = datetime.datetime.strptime(o[1], '%Y%m%d').date()
                underlying = o[0]
                tmp_params['t'] = (exp_date - val_date).days / 365.25
                tmp_params['future'] = f.futures_dict[o[0]]
                fn_ = r'C:/Users/PC2212/Desktop/논문/SOFR/Code/SOFR/results/%s/%s_%s.csv' % \
                      (val_date.strftime('%Y%m%d'), o[1], underlying)
                option_df = pandas.read_csv(fn_)[['Strike', 'Type', 'Last']]
                all_st = [(k, pc, mp) for k, pc, mp in zip(option_df['Strike'], option_df['Type'],
                                                                   option_df['Last'])]
                st = [(0.0, 'P'), (200.0, 'C')]
                for st_ in all_st:
                    st[0] = st_ if st_[1] == 'P' and st_[0] > st[0][0] else st[0]
                    st[1] = st_ if st_[1] == 'C' and st_[0] < st[1][0] else st[1]
                tmp_params['strike_types'] = st
                prices = []
                res, variables_ = calibration(tmp_params, test_variable, start_variable, increment)
                for st_ in st:
                    prices.append([r[st_][0] for r in res])
                    for r, v in zip(res, variables_):
                        confidence_interval_dict[(o[0], exp_date, st_[0], st_[1], v)] = r[st_][1]
                if not isinstance(options_system_df, pandas.DataFrame):
                    cols = ['Expiry', 'Underlying', 'Type', 'Strike', 'Price'] + variables_
                    options_system_df = pandas.DataFrame(columns=cols)
                options_system_df.loc[len(options_system_df)] = \
                    [exp_date, underlying, st[0][1], st[0][0], st[0][2]] + prices[0]
                options_system_df.loc[len(options_system_df)] = \
                    [exp_date, underlying, st[1][1], st[1][0], st[1][2]] + prices[1]
    options_system_df.to_csv('%s%s_%s_%s.csv' % (target_dir_, 'options_system',
                                                 test_variable[0], test_variable[1]), index=False)
    best_variable = variables_[0]
    best_error = 1e6
    for v in variables_:
        total_error = sum((abs(options_system_df[v] - options_system_df['Price']) - 0.005).clip(lower=0.0))
        if total_error < best_error:
            best_error = total_error
            best_variable = v
    print('best_variable')
    print(best_variable)
    print(best_error)
    best_result_df = copy.deepcopy(
        options_system_df[['Expiry', 'Underlying', 'Type', 'Strike', 'Price', best_variable]])
    best_result_df['Bid'] = best_result_df['Price'] - 0.0025
    best_result_df['Offer'] = best_result_df['Price'] + 0.0025
    best_result_df['Lower CI'] = best_result_df.apply(lambda row: confidence_interval_dict[
        (row['Underlying'], row['Expiry'], row['Strike'], row['Type'], best_variable)][0], axis=1)
    best_result_df['Upper CI'] = best_result_df.apply(lambda row: confidence_interval_dict[
        (row['Underlying'], row['Expiry'], row['Strike'], row['Type'], best_variable)][1], axis=1)
    best_result_df['Bid Vol'] = best_result_df.apply(lambda row: implied_vol_from_df_row(row, params, 'Bid', f), axis=1)
    best_result_df['Offer Vol'] = best_result_df.apply(lambda row: implied_vol_from_df_row(row, params, 'Offer', f),
                                                       axis=1)
    best_result_df['Lower CI Vol'] = best_result_df.apply(lambda row:
                                                          implied_vol_from_df_row(row, params, 'Lower CI', f), axis=1)
    best_result_df['Upper CI Vol'] = best_result_df.apply(lambda row:
                                                          implied_vol_from_df_row(row, params, 'Upper CI', f), axis=1)
    best_result_df.to_csv('%s%s.csv' % (target_dir_, 'best_result'), index=False)
    implied_vol_df = best_result_df.groupby(['Expiry', 'Underlying']). \
        agg({'Expiry': 'first', 'Bid Vol': 'mean', 'Offer Vol': 'mean', 'Lower CI Vol': 'mean', 'Upper CI Vol': 'mean'})
    implied_vol_df.to_csv('%s%s.csv' % (target_dir_, 'implied_vol_df'), index=False)

    plt.fill_between(implied_vol_df['Expiry'], implied_vol_df['Bid Vol'], implied_vol_df['Offer Vol'],
                     color='black', label='market bid/offer')
    plt.fill_between(implied_vol_df['Expiry'], implied_vol_df['Lower CI Vol'],
                     implied_vol_df['Upper CI Vol'], color='red', alpha=0.2, label='model ci=95%')
    plt.ylabel('implied normal vol (bp)')
    plt.xlabel('expiry')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

    return best_variable

    # raw_data_df['Type'] = raw_data_df.apply(lambda row: row['Strike'][-1], axis=1)


def calibrate_system_skew_2d(params, test_variables, underlying_list=None, expiry_months=None):
    print(test_variables)
    val_date = datetime.date(2023, 6, 16)
    st_date = datetime.date(2023, 3, 15)
    e_date = datetime.date(2024, 9, 17)
    f = FuturesSet(params['val_date'], e_date)
    fc = ForwardCurve(params['val_date'], st_date, e_date)
    fc.calibrate(f)
    options, target_dir_ = convert_option_data(params['val_date'], f.futures_price_dict)
    options_system_df = None
    confidence_interval_dict = {}
    variables_ = None
    for o in options:
        if not underlying_list or o[0] in underlying_list:
            exp_date = datetime.datetime.strptime(o[1], '%Y%m%d').date()
            if not expiry_months or exp_date.month in expiry_months:
                tmp_params = copy.deepcopy(params)
                print((o[0], o[1]))
                exp_date = datetime.datetime.strptime(o[1], '%Y%m%d').date()
                underlying = o[0]
                tmp_params['t'] = (exp_date - val_date).days / 365.25
                tmp_params['future'] = f.futures_dict[o[0]]
                fn_ = r'C:/Users/PC2212/Desktop/논문/SOFR/Code/SOFR/results/%s/%s_%s.csv' % \
                      (val_date.strftime('%Y%m%d'), o[1], underlying)
                option_df = pandas.read_csv(fn_)[['Strike', 'Type', 'Last']]
                all_st = [(k, pc, mp) for k, pc, mp in zip(option_df['Strike'], option_df['Type'],
                                                                   option_df['Last'])]
                st = []
                fut = f.futures_dict[underlying]
                atm = f.futures_price_dict[underlying]
                for st_ in all_st:
                    if abs(st_[0] - atm) < 1.0:
                        st.append(st_)
                #st = [(s_[0], 'P', s_[2]) for s_ in st]
                tmp_params['strike_types'] = st
                prices = []
                res, variables_1, variables_2 = calibration_2d(tmp_params, test_variables)
                if not isinstance(options_system_df, pandas.DataFrame):
                    cols = ['Expiry', 'Underlying', 'Type', 'Strike', 'Variable_1', 'Variable_2', 'Price', 'Bid', 'Ask',
                            'Vol', 'BidVol', 'AskVol', 'Model Price', 'LB Price', 'UB Price', 'Model Vol', 'LB Vol',
                            'UB Vol']
                    options_system_df = pandas.DataFrame(columns=cols)
                for r, v1, v2 in zip(res, variables_1, variables_2):
                    for r_k, r_v in r.items():
                        options_system_df.loc[len(options_system_df)] = \
                            [exp_date, underlying, r_k[1], r_k[0], v1, v2, r_k[2], r_k[2]-0.0025, r_k[2]+0.0025,
                             future_option_implied_vol(val_date, exp_date, fut, (r_k[1], r_k[0]), r_k[2]),
                             future_option_implied_vol(val_date, exp_date, fut, (r_k[1], r_k[0]), r_k[2]-0.0025),
                             future_option_implied_vol(val_date, exp_date, fut, (r_k[1], r_k[0]), r_k[2]+0.0025),
                             r_v[0], r_v[1][0], r_v[1][1],
                             future_option_implied_vol(val_date, exp_date, fut, (r_k[1], r_k[0]), r_v[0]),
                             future_option_implied_vol(val_date, exp_date, fut, (r_k[1], r_k[0]), r_v[1][0]),
                             future_option_implied_vol(val_date, exp_date, fut, (r_k[1], r_k[0]), r_v[1][1])]

    options_system_df.to_csv('%s%s.csv' % (target_dir_, 'options_system'))



def calibrate_system_skew(params, test_variable, start_variable, increment, underlying_list=None, expiry_months=None):
    print("test_varable", test_variable)
    val_date = datetime.date(2023, 6, 16)
    st_date = datetime.date(2023, 3, 15)
    e_date = datetime.date(2024, 9, 17)
    f = FuturesSet(params['val_date'], e_date)
    fc = ForwardCurve(params['val_date'], st_date, e_date)
    fc.calibrate(f)
    options, target_dir_ = convert_option_data(params['val_date'], f.futures_price_dict)
    options_system_df = None
    confidence_interval_dict = {}
    variables_ = None
    for o in options:
        if not underlying_list or o[0] in underlying_list:
            exp_date = datetime.datetime.strptime(o[1], '%Y%m%d').date()
            if not expiry_months or exp_date.month in expiry_months:
                tmp_params = copy.deepcopy(params)
                print((o[0], o[1]))
                exp_date = datetime.datetime.strptime(o[1], '%Y%m%d').date()
                underlying = o[0]
                tmp_params['t'] = (exp_date - val_date).days / 365.25
                tmp_params['future'] = f.futures_dict[o[0]]
                fn_ = r'C:/Users/PC2212/Desktop/논문/SOFR/Code/SOFR/results/%s/%s_%s.csv' % \
                      (val_date.strftime('%Y%m%d'), o[1], underlying)
                option_df = pandas.read_csv(fn_)[['Strike', 'Type', 'Last']]
                all_st = [(k, pc, mp) for k, pc, mp in zip(option_df['Strike'], option_df['Type'], option_df['Last'])]
                st = []
                fut = f.futures_dict[underlying]
                atm = f.futures_price_dict[underlying]
                for st_ in all_st:
                    if abs(st_[0] - atm) < 1.0:
                        st.append(st_)
                #st = [(s_[0], 'P', s_[2]) for s_ in st]
                tmp_params['strike_types'] = st
                prices = []
                res, variables_ = calibration(tmp_params, test_variable, start_variable, increment)
                if not isinstance(options_system_df, pandas.DataFrame):
                    cols = ['Expiry', 'Underlying', 'Type', 'Strike', 'Variable', 'Price', 'Bid', 'Ask', 'Vol',
                            'BidVol', 'AskVol', 'Model Price', 'LB Price', 'UB Price', 'Model Vol', 'LB Vol',
                            'UB Vol']
                    options_system_df = pandas.DataFrame(columns=cols)
                for r, v in zip(res, variables_):
                    for r_k, r_v in r.items():
                        options_system_df.loc[len(options_system_df)] = \
                            [exp_date, underlying, r_k[1], r_k[0], v, r_k[2], r_k[2]-0.0025, r_k[2]+0.0025,
                             future_option_implied_vol(val_date, exp_date, fut, (r_k[1], r_k[0]), r_k[2]),
                             future_option_implied_vol(val_date, exp_date, fut, (r_k[1], r_k[0]), r_k[2]-0.0025),
                             future_option_implied_vol(val_date, exp_date, fut, (r_k[1], r_k[0]), r_k[2]+0.0025),
                             r_v[0], r_v[1][0], r_v[1][1],
                             future_option_implied_vol(val_date, exp_date, fut, (r_k[1], r_k[0]), r_v[0]),
                             future_option_implied_vol(val_date, exp_date, fut, (r_k[1], r_k[0]), r_v[1][0]),
                             future_option_implied_vol(val_date, exp_date, fut, (r_k[1], r_k[0]), r_v[1][1])]
    if len(test_variable) == 2:
        options_system_df.to_csv('%s%s_%s_%s.csv' % (target_dir_, 'options_system',
                                                     test_variable[0], test_variable[1]), index=False)
    elif len(test_variable) == 3:
        options_system_df.to_csv('%s%s_%s_%s_%s.csv' % (target_dir_, 'options_system', test_variable[0],
                                                        test_variable[1], test_variable[2]), index=False)


def implied_vol_from_df_row(row, params, target, fut_set):
    return future_option_implied_vol(params['val_date'], row['Expiry'], fut_set.futures_dict[row['Underlying']],
                                     (row['Type'], row['Strike']), row[target]) * 1e4


def future_option_implied_vol(value_date, expiry, underlying, type_strike, price):
    t = (expiry - value_date).days / 365
    pc = 1 if type_strike[0] == 'P' else -1
    return implied_vol(t, pc, 0.01 * (100 - underlying.market_price), 0.01 * (100 - type_strike[1]), 0.01 * price)


def implied_vol(t, pc, s, k, target_price):
    max_iter = 200
    precision = 1e-5
    vol = 0.005
    for i in range(0, max_iter):
        price, vega = normal_price(t, pc, s, k, vol)
        diff = target_price - price
        if abs(diff) < precision:
            return vol
        vol = vol + diff / vega
    return vol


def normal_price(t, pc, s, k, vol):
    m = (s - k) * pc
    d = m / (numpy.sqrt(t) * vol)
    vega = numpy.sqrt(t) * norm.pdf(d)
    return m * norm.cdf(d) + vol * vega, vega


def drift_impact(params, futures_set, fut_start, n_futures, max_threads):
    futures = ['slv20', 'slx20', 'slz20', 'slf21', 'slg21', 'slh21', 'slj21', 'slk21', 'slm21', 'sln21', 'slq21',
               'slu21', 'slv21', 'squ20', 'sqz20', 'sqh21', 'sqm21', 'squ21', 'sqz21', 'sqh22', 'sqm22']
    expiries = [datetime.date(2020, 10, 31), datetime.date(2020, 11, 30), datetime.date(2020, 12, 31),
                datetime.date(2021, 1, 31), datetime.date(2021, 2, 28), datetime.date(2021, 3, 31),
                datetime.date(2021, 4, 30), datetime.date(2021, 5, 31), datetime.date(2021, 6, 30),
                datetime.date(2021, 7, 31), datetime.date(2021, 8, 31), datetime.date(2021, 9, 30),
                datetime.date(2021, 10, 31), datetime.date(2020, 12, 15), datetime.date(2021, 3, 15),
                datetime.date(2021, 6, 15), datetime.date(2021, 9, 15), datetime.date(2021, 12, 15),
                datetime.date(2022, 3, 15), datetime.date(2022, 6, 15), datetime.date(2022, 9, 15)]
    t_s = [(e - params['val_date']).days / 365.25 for e in expiries]
    arg_list = []
    for e, f_ in zip(t_s, futures):
        tmp_args = copy.deepcopy(params)
        tmp_args['t'] = e
        tmp_args['future'] = futures_set.futures_dict[f_]
        arg_list.append(tmp_args)
    parallel_runner_wrap(arg_list[0])
    with Pool(max(len(arg_list), max_threads)) as p:
        results = p.map(parallel_runner_wrap, arg_list[fut_start:fut_start+n_futures])
    res = numpy.array([[r[0.0, 'C'][0], r[0.0, 'C'][1][0], r[0.0, 'C'][1][1]] for r in results])
    numpy.savetxt('drift_check.csv', res, delimiter=',')


def calibrate_skew():
    start_date = datetime.date(2023, 3, 15)
    end_date = datetime.date(2024, 9, 17)
    val_date = datetime.date(2023, 6, 16)
    fut_set = FuturesSet(val_date, end_date)
    f_curve = ForwardCurve(val_date, start_date, end_date)
    f_curve.calibrate(fut_set)
    convert_option_data(val_date, fut_set.futures_price_dict)
    expiry_date = datetime.date(2023, 6, 21)
    t_ = (expiry_date - val_date).days / 365.25
    params_ = {'paths': 100000,
               'steps': 50,
               'factors': 3,
               't': t_,
               'f_t': 0.0,
               'lambdas': [0.16961, 0.297837, 0.0456138],
               'val_date': val_date,
               'corr': [-0.033, -0.53962, -0.43],
               'alpha': [([0.0001, 0.0001, 0.0001, 0.0001], [0.2, 0.45, 0.7]),
                         ([0.0001, 0.0001, 0.0001, 0.0001], [0.2, 0.45, 0.7]),
                         ([0.0001, 0.0001, 0.0001, 0.0001], [0.2, 0.45, 0.7])],
               'theta': [12.002, 6.0000599999999995, 6.0000599999999995],
               'm_dates': f_curve.get_meeting_dates_yf(),
               'gammas': gamma(),
               'factor_vols': [0.02338, 0.0100156, 0.000144],
               'label': 'original',
               'future': fut_set.futures_dict['SFRM23'],
               'strike_types': [(0.0, 'C')],
               'drift': False}

    fut_list = ['SFRU23', 'SFRZ23', 'SFRH24', 'SFRM24']

    exp_months = [3, 6, 9, 12]

    test_params = [(('alpha', 0, 0), 3.13, 0.002),
                   (('alpha', 0, 1), 1.3, 0.02),
                   (('alpha', 0, 2), 3.3, 0.02) ]
                   #(('alpha', 0, 3), 0.8, 0.02),
                   #(('alpha', 1, 0), 0.75, 0.01),
                   #(('alpha', 1, 1), 0.6, 0.01),
                #    (('alpha', 1, 2), 0.6, 0.01),
                #    (('alpha', 1, 3), 0.1, 0.02),
                #    (('alpha', 2, 0), 2.3, 0.1),
                #    (('alpha', 2, 1), 0.0, 0.1),
                #    (('alpha', 2, 2), 0.0, 0.1),
                #    (('alpha', 2, 3), 2.5, 0.1)
                #     ]

    for tp in test_params:
        calibrate_system_skew(params_, tp[0], tp[1], tp[2], fut_list, exp_months)


def calibrate_skew_2d():
    start_date = datetime.date(2023, 3, 15)
    end_date = datetime.date(2024, 9, 17)
    val_date = datetime.date(2023, 6, 16)
    fut_set = FuturesSet(val_date, end_date)
    f_curve = ForwardCurve(val_date, start_date, end_date)
    f_curve.calibrate(fut_set)
    convert_option_data(val_date, fut_set.futures_price_dict)
    expiry_date = datetime.date(2023, 6, 21)
    t_ = (expiry_date - val_date).days / 365.25
    params_ = {'paths': 500000,
               'steps': 50,
               'factors': 3,
               't': t_,
               'f_t': 0.0,
               'lambdas': [0.16961, 0.297837, 0.0456138],
               'val_date': val_date,
               'corr': [-0.033, -0.53962, -0.43],
               'alpha': [-0.1, 0.07075, -1.06],
               'theta': [2.43, 1.18, 3.03],
               'm_dates': f_curve.get_meeting_dates_yf(),
               'gammas': gamma(),
               'factor_vols': [0.02338, 0.0100156, 0.000144], 
               'label': 'original',
               'future': fut_set.futures_dict['SFRM23'],
               'strike_types': [(0.0, 'C')],
               'drift': False}

    fut_list = ['SFRU23', 'SFRZ23', 'SFRH24', 'SFRM24']

    exp_months = [3, 6, 9, 12]
    test_params = [(('factor_vols', 0), 0.0233, 0.00005),
                   (('factor_vols', 1), 0.010015, 0.00005),
                   (('factor_vols', 2), 0.00014, 0.00005),
                   (('lambdas', 0), 0.167, 0.002),
                   (('lambdas', 1), 0.295, 0.002),
                   (('lambdas', 2), 0.043, 0.002),
                   (('corr', 0), -0.048, 0.005),
                   (('corr', 1), -0.54, 0.005),
                   (('corr', 2), -0.435, 0.005),
                   (('alpha', 0), -0.11, 0.01),
                   (('alpha', 1), 0.06, 0.01),
                   (('alpha', 2), -1.07, 0.01),
                   (('theta', 0), 2.41, 0.02),
                   (('theta', 1), 1.16, 0.02),
                   (('theta', 2), 3.01, 0.02)]
    
    # test_params에서 2개씩 짝지어 보정 수행
    for i in range(len(test_params)):
        for j in range(i + 1, len(test_params)):
            param_pair = [test_params[i], test_params[j]]
            print(f"Running calibration for {param_pair}")
            calibrate_system_skew_2d(params_, param_pair, fut_list, exp_months)

    '''        
    # test_params에서 3개씩 짝지어 보정 수행
    for i in range(len(test_params)):
        for j in range(i + 1, len(test_params)):
            for k in range(j + 1, len(test_params)):
                param_triple = [test_params[i], test_params[j], test_params[k]]
                print(f"Running calibration for {param_triple}")
                calibrate_system_skew_2d(params_, param_triple, fut_list, exp_months)
    '''


    #calibrate_system_skew_2d(params_, test_params, fut_list, exp_months)

'''
'alpha': [([3.1, 1.3, 3.2, 0.5], [0.2, 0.45, 0.7]),
                         ([0.76, 0.76, 0.62, 0.52], [0.2, 0.45, 0.7]),
                         ([3.0, 1.2, 3.0, 4.2], [0.2, 0.45, 0.7])],
'''

##########################################################################################################
'''
def run_calibration():
    start_date = datetime.date(2023, 3, 15)  # !!! 'future'의 선물코드의 reference_start !!!
    end_date = datetime.date(2024, 9, 17)  
    val_date = datetime.date(2023, 6, 16)  # !!! 'future'의 선물코드의 reference_end - 5 !!!
    fut_set = FuturesSet(val_date, end_date)
    f_curve = ForwardCurve(val_date, start_date, end_date)
    f_curve.calibrate(fut_set)
    convert_option_data(val_date, fut_set.futures_price_dict)
    expiry_date = datetime.date(2023, 6, 21)  # !!! 'future'의 선물코드의 reference_end !!!
    t_ = (expiry_date - val_date).days / 365.25

    params_ = {'paths': 500000,
                'steps': 50,
                'factors': 3,
                't': t_,
                'f_t': 0.0,
                'lambdas': [0.01, 0.41, 4.5], 
                'val_date': val_date,
                'corr': [0.0, 0.0, 0.6], # -1 < corr < 1
                'alpha': [0.4, 0.4, 2.4],  
                'theta': [0.0, 0.0, 0.001], 
                'm_dates': f_curve.get_meeting_dates_yf(),
                'gammas': gamma(),
                'factor_vols': [0.0095, 0.005, 0.008],
                'label': 'original',
                'future': fut_set.futures_dict['SFRM23'],  # !!! fut_list_liquid[0]보다 앞의 계약 !!!
                'strike_types': [(0.0, 'C')],
                'drift': True}

    fut_list_liquid = ['SFRU23', 'SFRZ23', 'SFRH24', 'SFRM24']
    fut_list = fut_list_liquid
    exp_months = [3, 6, 9, 12]

    calibrate_system(params_, ('factor_vols', 2), 0.008, 0.0005, fut_list)


    # Lambda와 factor_vols를 최적화하는 범위 설정
    factor_vol_range = [0.004, 0.1]
    lambda_range = [4.0, 15.0]

    #calibrate_system_bisection(params_, ['alpha', 'theta'], [0, 0], alpha_range, theta_range, exp_months, fut_list_liquid)


    #calibrate_system_2(params_, ['factor_vols', 'lambdas'], [2, 2], factor_vol_range, lambda_range, exp_months, fut_list_liquid)

'''


'''
calibrate_system(params_, ('factor_vols', 2), 0.0, 0.01,
                     ['squ22', 'sqz22', 'sqh23', 'sqm23', 'squ23', 'sqz23', 'sqh24', 'sqm24'])

calibrate_system_bisection(params_, ('factor_vols', 'factor_vols'), (0, 2), (0.015, 0.02), (0.0, 0.01), [3, 6, 9, 12],
                               ['squ22', 'sqz22', 'sqh23', 'sqm23', 'squ23', 'sqz23', 'sqh24', 'sqm24'])

calibrate_system_2(params_, ('lambdas', 0), 0.0, 0.01,
                       ['squ22', 'sqz22', 'sqh23', 'sqm23', 'squ23', 'sqz23', 'sqh24', 'sqm24'])

'''

#######################################################################################################################################################
'''
if __name__ == '__main__':
    t0_ = time.time()
    #sofr_history()
    #sofr_variance()
    #sofr_qq_plot()
    #chart_calibration_results('skew_calib_original.csv', 'skew_calib_original.pdf')
    #chart_calibration_results('skew_calib_timedep.csv', 'skew_calib_timedep.pdf')
    #chart_accrual_period()
    run_calibration()
    #sofr_qq_plot()
    #factor_sensitivity_charts()
    #calibrate_skew()
    #calibrate_skew_2d()
    #sofr_qq_plot()
    #sofr_variance()
    #sofr_history()
    #chart_gamma_lambda_comparison()
    # run_drift_impact(zero_vol=False, drift=True, fut_start=20, n_futures=1, paths=2000000, max_threads=4, directory='')
    t1_ = time.time()
    print(t1_ - t0_)
'''

#########################################################################################################################
'''

def implied_vol_from_df_row_JW(row, params, target, fut_set):
    return future_option_implied_vol(params['val_date'], row['Expiry'], fut_set.futures_dict[row['Underlying']],
                                     (row['Type'], row['Strike']), row[target]) * 1e4


def future_option_implied_vol_JW(value_date, expiry, underlying, type_strike, price):
    t = (expiry - value_date).days / 365
    pc = 1 if type_strike[0] == 'P' else -1
    return implied_vol(t, pc, 0.01 * (100 - underlying.market_price), 0.01 * (100 - type_strike[1]), 0.01 * price)


def implied_vol_JW(t, pc, s, k, target_price):
    max_iter = 200
    precision = 1e-5
    vol = 0.005
    for i in range(0, max_iter):
        price, vega = normal_price(t, pc, s, k, vol)
        if vega == 0:  # vega가 0일 경우 오류 발생 방지 및 건너뛰기
            print(f"Skipping calculation due to zero vega (strike: {k}, spot: {s}), price: {target_price})")
            return float('nan')  # None을 반환해서 건너뛰도록 처리
        diff = target_price - price
        if abs(diff) < precision:
            return vol
        vol = vol + diff / vega
    return vol


def normal_price_JW(t, pc, s, k, vol):
    m = (s - k) * pc
    d = m / (numpy.sqrt(t) * vol)
    vega = numpy.sqrt(t) * norm.pdf(d)
    return m * norm.cdf(d) + vol * vega, vega


def calibrationJW(args, scan_variable, scan_start, scan_inc, parallel=True):
    arg_list = []
    n = 6
    variables = [scan_start + i * scan_inc for i in range(n)]
    for v in variables:
        tmp_args = copy.deepcopy(args)
        if len(scan_variable) == 2:
            tmp_args[scan_variable[0]][scan_variable[1]] = v
        elif len(scan_variable) == 3:
            tmp_args[scan_variable[0]][scan_variable[1]][0][scan_variable[2]] = v
        arg_list.append(tmp_args)
    if parallel:
        with Pool(max(len(arg_list), n)) as p:
            return p.map(parallel_runner_wrap, arg_list), variables
    else:
        return [Runner(**a).get_option_results(a['strike_types']) for a in arg_list], variables


def calibrate_systemJW(params, test_variable, start_variable, increment, underlying_list=None, expiry_months=None):
    val_date = datetime.date(2023, 6, 16)
    st_date = datetime.date(2023, 3, 15)
    e_date = datetime.date(2024, 9, 17)
    f = FuturesSet(params['val_date'], e_date)
    fc = ForwardCurve(params['val_date'], st_date, e_date)
    fc.calibrate(f)
    options, target_dir_ = convert_option_data(params['val_date'], f.futures_price_dict)
    options_system_df = None
    confidence_interval_dict = {}
    variables_ = None
    for o in options:
        if not underlying_list or o[0] in underlying_list:
            exp_date = datetime.datetime.strptime(o[1], '%Y%m%d').date()
            if not expiry_months or exp_date.month in expiry_months:
                tmp_params = copy.deepcopy(params)
                print((o[0], o[1]))
                exp_date = datetime.datetime.strptime(o[1], '%Y%m%d').date()
                underlying = o[0]
                tmp_params['t'] = (exp_date - val_date).days / 365.25
                tmp_params['future'] = f.futures_dict[o[0]]
                fn_ = r'C:/Users/PC2212/Desktop/논문/SOFR/Code/SOFR/results/%s/%s_%s.csv' % \
                      (val_date.strftime('%Y%m%d'), o[1], underlying)
                option_df = pandas.read_csv(fn_)[['Strike', 'Type', 'Last']]

                # 모든 행사가를 사용하여 옵션 가격 계산
                tmp_params['strike_types'] = [(row['Strike'], row['Type']) for _, row in option_df.iterrows()]
                prices = []
                res, variables_ = calibrationJW(tmp_params, test_variable, start_variable, increment)
                
                # 여기에 calibrationJW에 get_options 대신 get_results를 써어 res를 받기. res를 저장하는 csv는 s,k,t,pc,market price에 대한 데이터프레임이여야 

                for st_ in tmp_params['strike_types']:
                    prices.append([r[st_][0] for r in res])
                    for r, v in zip(res, variables_):
                        confidence_interval_dict[(o[0], exp_date, st_[0], st_[1], v)] = r[st_][1]

                if not isinstance(options_system_df, pandas.DataFrame):
                    cols = ['Expiry', 'Underlying', 'Type', 'Strike', 'Price'] + variables_
                    options_system_df = pandas.DataFrame(columns=cols)

                for i, st_ in enumerate(tmp_params['strike_types']):
                    row_data = [exp_date, underlying, st_[1], st_[0], prices[i][0]]
                    row_data += prices[i][1:]

                    # 만약 prices[i][1:]에 이미 variables_ 값이 포함되어 있다면, 따로 추가하지 않음
                    if len(row_data) < len(options_system_df.columns):  # 열 개수 부족할 경우만 추가
                        row_data += variables_[:len(options_system_df.columns) - len(row_data)]  # 부족한 값만큼 추가

                    #print(f"row_data length: {len(row_data)}, options_system_df columns length: {len(options_system_df.columns)}")
                    
                    if len(row_data) == len(options_system_df.columns):  # 열 개수 맞는지 확인
                        options_system_df.loc[len(options_system_df)] = row_data
                    else:
                        print(f"Error: Mismatched columns for {underlying}, strike: {st_[0]}")
    
    options_system_df.to_csv('%s%s_%s_%s.csv' % (target_dir_, 'options_system',
                                                 test_variable[0], test_variable[1]), index=False)
    
    best_variable = variables_[0]
    best_error = 1e6
    for v in variables_:
        total_error = sum((abs(options_system_df[v] - options_system_df['Price']) - 0.005).clip(lower=0.0))
        if total_error < best_error:
            best_error = total_error
            best_variable = v
    print('best_variable')
    print(best_variable)
    print(best_error)

    best_result_df = copy.deepcopy(
        options_system_df[['Expiry', 'Underlying', 'Type', 'Strike', 'Price', best_variable]])
    best_result_df['Bid'] = best_result_df['Price'] - 0.0025
    best_result_df['Offer'] = best_result_df['Price'] + 0.0025
    best_result_df['Lower CI'] = best_result_df.apply(lambda row: confidence_interval_dict[
        (row['Underlying'], row['Expiry'], row['Strike'], row['Type'], best_variable)][0], axis=1)
    best_result_df['Upper CI'] = best_result_df.apply(lambda row: confidence_interval_dict[
        (row['Underlying'], row['Expiry'], row['Strike'], row['Type'], best_variable)][1], axis=1)
    
    best_result_df['Bid Vol'] = best_result_df.apply(lambda row: implied_vol_from_df_row(row, params, 'Bid', f) 
                                                     if implied_vol_from_df_row(row, params, 'Bid', f) is not None else float('nan'), axis=1)

    best_result_df['Offer Vol'] = best_result_df.apply(lambda row: implied_vol_from_df_row(row, params, 'Offer', f)
                                                       if implied_vol_from_df_row(row, params, 'Offer', f) is not None else float('nan'), axis=1)
    best_result_df['Lower CI Vol'] = best_result_df.apply(lambda row:
                                                          implied_vol_from_df_row(row, params, 'Lower CI', f)
                                                          if implied_vol_from_df_row(row, params, 'Lower CI', f) is not None else float('nan'), axis=1)
    best_result_df['Upper CI Vol'] = best_result_df.apply(lambda row:
                                                          implied_vol_from_df_row(row, params, 'Upper CI', f)
                                                          if implied_vol_from_df_row(row, params, 'Upper CI', f) is not None else float('nan'), axis=1)
    best_result_df.to_csv('%s%s.csv' % (target_dir_, 'best_result'), index=False)
    implied_vol_df = best_result_df.groupby(['Expiry', 'Underlying']). \
        agg({'Expiry': 'first', 'Bid Vol': 'mean', 'Offer Vol': 'mean', 'Lower CI Vol': 'mean', 'Upper CI Vol': 'mean'})
    implied_vol_df.to_csv('%s%s.csv' % (target_dir_, 'implied_vol_df'), index=False)

    plt.fill_between(implied_vol_df['Expiry'], implied_vol_df['Bid Vol'], implied_vol_df['Offer Vol'],
                     color='black', label='market bid/offer')
    plt.fill_between(implied_vol_df['Expiry'], implied_vol_df['Lower CI Vol'],
                     implied_vol_df['Upper CI Vol'], color='red', alpha=0.2, label='model ci=95%')
    plt.ylabel('implied normal vol (bp)')
    plt.xlabel('expiry')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

    return best_variable



def run_calibration():
    start_date = datetime.date(2023, 3, 15)  # !!! 'future'의 선물코드의 reference_start !!!
    end_date = datetime.date(2024, 9, 17)  
    val_date = datetime.date(2023, 6, 16)  # !!! 'future'의 선물코드의 reference_end - 5 !!!
    fut_set = FuturesSet(val_date, end_date)
    f_curve = ForwardCurve(val_date, start_date, end_date)
    f_curve.calibrate(fut_set)
    convert_option_data(val_date, fut_set.futures_price_dict)
    expiry_date = datetime.date(2023, 6, 21)  # !!! 'future'의 선물코드의 reference_end !!!
    t_ = (expiry_date - val_date).days / 365.25
    params_ = {'paths': 100000,
               'steps': 10,
               'factors': 3,
               't': t_,
               'f_t': 0.0,
               'lambdas': [0.16961, 0.297837, 0.0456138], # 높일수록 우상향
               'val_date': val_date,
               'corr': [-0.033, -0.53962, -0.43], # -1 < corr < 1
               'alpha': [0.0001, 0.0001, 0.0001], # 그래프의 높이 
               'theta': [12.002, 6.0000599999999995, 6.0000599999999995], # 높일수록 완만한 우상향 & 일정 값에 수렴
               'm_dates': f_curve.get_meeting_dates_yf(),
               'gammas': gamma(),
               'factor_vols': [0.02338, 0.0100156, 0.000144], # 그래프의 높이
               'label': 'original',
               'future': fut_set.futures_dict['SFRM23'],  # !!! fut_list_liquid[0]보다 앞의 계약 !!!
               'strike_types': [(0.0, 'C')],
               'drift': True}

    fut_list_liquid = ['SFRU23', 'SFRZ23', 'SFRH24', 'SFRM24']

    exp_months = [3, 6, 9, 12]

    calibrate_systemJW(params_, ('alpha', 0), 0.0001, 0.005, fut_list_liquid)


    #calibrate_system(params_, ('factor_vols', 2), 0.0, 0.01,
                     #['squ22', 'sqz22', 'sqh23', 'sqm23', 'squ23', 'sqz23', 'sqh24', 'sqm24'])

if __name__ == '__main__':
    t0_ = time.time()
    #sofr_history()
    #sofr_variance()
    #sofr_qq_plot()
    #chart_calibration_results('skew_calib_original.csv', 'skew_calib_original.pdf')
    #chart_calibration_results('skew_calib_timedep.csv', 'skew_calib_timedep.pdf')
    #chart_accrual_period()
    run_calibration()
    #sofr_qq_plot()
    #factor_sensitivity_charts()
    #calibrate_skew()
    #calibrate_skew_2d()
    #sofr_qq_plot()
    #sofr_variance()
    #sofr_history()
    #chart_gamma_lambda_comparison()
    # run_drift_impact(zero_vol=False, drift=True, fut_start=20, n_futures=1, paths=2000000, max_threads=4, directory='')
    t1_ = time.time()
    print(t1_ - t0_)
'''
#################################모든 행사가의 모델 옵션가격 저장, IV는 코랩에서 그렸음###################################################################



def calibrationJW(args, scan_variable, scan_start, scan_inc, parallel=True):
    arg_list = []
    n = 6
    variables = [scan_start + i * scan_inc for i in range(n)]
    for v in variables:
        tmp_args = copy.deepcopy(args)
        if len(scan_variable) == 2:
            tmp_args[scan_variable[0]][scan_variable[1]] = v
        elif len(scan_variable) == 3:
            tmp_args[scan_variable[0]][scan_variable[1]][0][scan_variable[2]] = v
        arg_list.append(tmp_args)
    if parallel:
        with Pool(max(len(arg_list), n)) as p:
            return p.map(parallel_runner_wrap, arg_list), variables
    else:
        return [Runner(**a).get_option_results(a['strike_types']) for a in arg_list], variables


def calibrate_systemJW(params, test_variable, start_variable, increment, underlying_list=None, expiry_months=None):
    val_date = datetime.date(2023, 6, 16)
    st_date = datetime.date(2023, 3, 15)
    e_date = datetime.date(2024, 9, 17)
    f = FuturesSet(params['val_date'], e_date)
    fc = ForwardCurve(params['val_date'], st_date, e_date)
    fc.calibrate(f)
    options, target_dir_ = convert_option_data(params['val_date'], f.futures_price_dict)
    options_system_df = None
    confidence_interval_dict = {}
    variables_ = None
    for o in options:
        if not underlying_list or o[0] in underlying_list:
            exp_date = datetime.datetime.strptime(o[1], '%Y%m%d').date()
            if not expiry_months or exp_date.month in expiry_months:
                tmp_params = copy.deepcopy(params)
                print((o[0], o[1]))
                exp_date = datetime.datetime.strptime(o[1], '%Y%m%d').date()
                underlying = o[0]
                tmp_params['t'] = (exp_date - val_date).days / 365.25
                tmp_params['future'] = f.futures_dict[o[0]]
                fn_ = r'C:/Users/PC2212/Desktop/논문/SOFR/Code/SOFR/results/%s/%s_%s.csv' % \
                      (val_date.strftime('%Y%m%d'), o[1], underlying)
                option_df = pandas.read_csv(fn_)[['Strike', 'Type', 'Last']]

                # 모든 행사가를 사용하여 옵션 가격 계산
                tmp_params['strike_types'] = [(row['Strike'], row['Type']) for _, row in option_df.iterrows()]
                prices = []
                res, variables_ = calibrationJW(tmp_params, test_variable, start_variable, increment)
                
                for st_ in tmp_params['strike_types']:
                    prices.append([r[st_][0] for r in res])
                    for r, v in zip(res, variables_):
                        confidence_interval_dict[(o[0], exp_date, st_[0], st_[1], v)] = r[st_][1]

                if not isinstance(options_system_df, pandas.DataFrame):
                    cols = ['Expiry', 'Underlying', 'Type', 'Strike', 'Price'] + variables_
                    options_system_df = pandas.DataFrame(columns=cols)

                for i, st_ in enumerate(tmp_params['strike_types']):
                    row_data = [exp_date, underlying, st_[1], st_[0], prices[i][0]]
                    row_data += prices[i][1:]

                    # 만약 prices[i][1:]에 이미 variables_ 값이 포함되어 있다면, 따로 추가하지 않음
                    if len(row_data) < len(options_system_df.columns):  # 열 개수 부족할 경우만 추가
                        row_data += variables_[:len(options_system_df.columns) - len(row_data)]  # 부족한 값만큼 추가

                    #print(f"row_data length: {len(row_data)}, options_system_df columns length: {len(options_system_df.columns)}")
                    
                    if len(row_data) == len(options_system_df.columns):  # 열 개수 맞는지 확인
                        options_system_df.loc[len(options_system_df)] = row_data
                    else:
                        print(f"Error: Mismatched columns for {underlying}, strike: {st_[0]}")
    
    if len(test_variable) == 2:
        options_system_df.to_csv('%s%s_%s_%s.csv' % (target_dir_, 'options_system',
                                                     test_variable[0], test_variable[1]), index=False)
    elif len(test_variable) == 3:
        options_system_df.to_csv('%s%s_%s_%s_%s.csv' % (target_dir_, 'options_system', test_variable[0],
                                                        test_variable[1], test_variable[2]), index=False)
    
    best_variable = variables_[0]
    best_error = 1e6
    for v in variables_:
        total_error = sum((abs(options_system_df[v] - options_system_df['Price']) - 0.005).clip(lower=0.0))
        if total_error < best_error:
            best_error = total_error
            best_variable = v
    print('best_variable')
    print(best_variable)
    print(best_error)
    return best_variable


def run_calibrationJW():
    start_date = datetime.date(2023, 3, 15)  # !!! 'future'의 선물코드의 reference_start !!!
    end_date = datetime.date(2024, 9, 17)  
    val_date = datetime.date(2023, 6, 16)  # !!! 'future'의 선물코드의 reference_end - 5 !!!
    fut_set = FuturesSet(val_date, end_date)
    f_curve = ForwardCurve(val_date, start_date, end_date)
    f_curve.calibrate(fut_set)
    convert_option_data(val_date, fut_set.futures_price_dict)
    expiry_date = datetime.date(2023, 6, 21)  # !!! 'future'의 선물코드의 reference_end !!!
    t_ = (expiry_date - val_date).days / 365.25
    # params_ = {'paths': 100000,
    #            'steps': 10,
    #            'factors': 3,
    #            't': t_,
    #            'f_t': 0.0,
    #            'lambdas': [0.01, 0.41, 4.5],
    #            'val_date': val_date,
    #            'corr': [0.0, 0.0, 0.6], # -1 < corr < 1
    #         #    'alpha': [([0.4, 0.4, 1.5, 0.4], [0.25, 0.50, 0.75]),
    #         #              ([4.6, 0.4, 0.4, 0.4], [0.25, 0.50, 0.75]), 
    #         #              ([4.3, 4.0, 0.4, 2.0], [0.25, 0.50, 0.75])], 
    #            'alpha': [0.4, 0.4, 2.4],
    #            'theta': [0.0, 0.0, 0.001], 
    #            'm_dates': f_curve.get_meeting_dates_yf(),
    #            'gammas': gamma(),
    #            'factor_vols': [0.0095, 0.005, 0.008],
    #            'label': 'original',
    #            'future': fut_set.futures_dict['SFRM23'],  # !!! fut_list_liquid[0]보다 앞의 계약 !!!
    #            'strike_types': [(0.0, 'C')],
    #            'drift': True}

    params_ = {'paths': 100000,
               'steps': 10,
               'factors': 3,
               't': t_,
               'f_t': 0.0,
               'lambdas': [0.01, 0.41, 4.5],
               'val_date': val_date,
               'corr': [0.0, 0.0, 0.6], # -1 < corr < 1
            #    'alpha': [([0.42, 0.45, 1.5, 0.41], [0.25, 0.50, 0.75]),
            #              ([4.6, 0.42, 0.38, 0.39], [0.25, 0.50, 0.75]), 
            #              ([4.3, 4.02, 0.41, 1.98], [0.25, 0.50, 0.75])],
               'alpha' : [0.42, 0.411, 2.4],
               'theta': [0.0, 0.0, 0.001], 
               'm_dates': f_curve.get_meeting_dates_yf(),
               'gammas': gamma(),
               'factor_vols': [0.0095, 0.005, 0.008],
               'label': 'original',
               'future': fut_set.futures_dict['SFRM23'],  # !!! fut_list_liquid[0]보다 앞의 계약 !!!
               'strike_types': [(0.0, 'C')],
               'drift': True}

    fut_list_liquid = ['SFRU23', 'SFRZ23', 'SFRH24', 'SFRM24']

    exp_months = [3, 6, 9, 12]


    # for i in range(100):
    #         print('lambdas_0')
    #         start = max(params_['lambdas'][0] - 0.005, 0.01)
    #         best = calibrate_systemJW(params_, ('lambdas', 0), start, 0.01, fut_list_liquid, exp_months)
    #         params_['lambdas'][0] = best
    #         print('corr_0')
    #         start = max(params_['corr'][0] - 0.1, -0.9)
    #         best = calibrate_systemJW(params_, ('corr', 0), start, 0.05, fut_list_liquid, exp_months)
    #         params_['corr'][0] = best
    #         print('corr_1')
    #         start = max(params_['corr'][1] - 0.1, -0.9)
    #         best = calibrate_systemJW(params_, ('corr', 1), start, 0.05, fut_list_liquid, exp_months)
    #         params_['corr'][1] = best

    #         print(params_['lambdas'])
    #         print(params_['corr'])


    calibrate_systemJW(params_, ('factor_vols', 0), 0.0095, 0.0003, fut_list_liquid)
    calibrate_systemJW(params_, ('factor_vols', 1), 0.005, 0.0003, fut_list_liquid)
    calibrate_systemJW(params_, ('factor_vols', 2), 0.008, 0.0003, fut_list_liquid)

    calibrate_systemJW(params_, ('lambdas', 0), 0.01, 0.01, fut_list_liquid)
    calibrate_systemJW(params_, ('lambdas', 1), 0.41, 0.005, fut_list_liquid)
    calibrate_systemJW(params_, ('lambdas', 2), 4.5, 0.005, fut_list_liquid)

    calibrate_systemJW(params_, ('alpha', 0), 0.42, 0.03, fut_list_liquid)
    calibrate_systemJW(params_, ('alpha', 1), 0.411, 0.03, fut_list_liquid)
    calibrate_systemJW(params_, ('alpha', 2), 2.4, 0.03, fut_list_liquid)

    calibrate_systemJW(params_, ('theta', 0), 0.0, 0.01, fut_list_liquid)
    calibrate_systemJW(params_, ('theta', 1), 0.0, 0.01, fut_list_liquid)
    calibrate_systemJW(params_, ('theta', 2), 0.6, 0.01, fut_list_liquid)

    # calibrate_systemJW(params_, ('corr', 0), -0.8, 0.1, fut_list_liquid) -0.6으로 수정됨
    # calibrate_systemJW(params_, ('corr', 1), -0.8, 0.1, fut_list_liquid) 0.6으로 수정됨
    # calibrate_systemJW(params_, ('corr', 2), 0.6, -0.05, fut_list_liquid)
    ######################################################################
    # calibrate_systemJW(params_, ('alpha', 0, 0), 0.4, 0.01, fut_list_liquid)
    # calibrate_systemJW(params_, ('alpha', 0, 1), 0.4, 0.01, fut_list_liquid)
    # calibrate_systemJW(params_, ('alpha', 0, 2), 0.4, 0.2, fut_list_liquid) 3.2로 수정됨
    # calibrate_systemJW(params_, ('alpha', 0, 3), 0.4, 0.01, fut_list_liquid)

    # calibrate_systemJW(params_, ('alpha', 1, 0), 4.6, 0.01, fut_list_liquid)
    # calibrate_systemJW(params_, ('alpha', 1, 1), 0.4, 0.01, fut_list_liquid)
    # calibrate_systemJW(params_, ('alpha', 1, 2), 0.4, 0.01, fut_list_liquid)
    # calibrate_systemJW(params_, ('alpha', 1, 3), 0.4, 0.01, fut_list_liquid)

    # calibrate_systemJW(params_, ('alpha', 2, 0), 4.3, 0.01, fut_list_liquid)
    # calibrate_systemJW(params_, ('alpha', 2, 1), 4.0, 0.01, fut_list_liquid)
    # calibrate_systemJW(params_, ('alpha', 2, 2), 0.4, 0.01, fut_list_liquid)
    # calibrate_systemJW(params_, ('alpha', 2, 3), 2.0, 0.01, fut_list_liquid)



if __name__ == '__main__':
    t0_ = time.time()
    run_calibrationJW()
    t1_ = time.time()
    print(t1_ - t0_)


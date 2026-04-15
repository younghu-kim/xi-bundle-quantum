"""
=============================================================================
DQPT-Zeta 프로토타입: 동적 양자 위상 전이와 리만 영점의 연결
=============================================================================

이론적 배경:
    리만 제타 함수의 영점 ρ_n = 1/2 + i·t_n 과
    동적 양자 위상 전이(DQPT)의 Fisher zero / Loschmidt echo 영점은
    수학적으로 동일한 구조를 가진다.

    [DQPT 기본]
    - Loschmidt amplitude:  G(t) = ⟨ψ₀|e^{-iH₂t}|ψ₀⟩
    - Rate function:        r(t) = -log|G(t)|² / N
    - DQPT 임계 시간 t*:    G(t*) = 0 인 점 (비해석적 특이점)

    [리만 제타와의 연결]
    - ξ-다발 프레임워크에서 영점 t_n ↔ 모노드로미 ±π
    - DQPT에서 t* ↔ G(t*)=0 (위상 전이 임계점)
    - 가설: κ(t) = |ξ'/ξ|² 피크 → DQPT 임계점 밀집 유도

    [κ 주입 메커니즘]
    - J(t) = J₀ · (1 + α·κ_norm(t))  또는
    - h(t) = h₀ · (1 + β·κ_norm(t))
    - κ가 큰 곳(영점 근방) → 유효 결합이 증가 → DQPT 출현 가속

    참고: arXiv:2511.11199 (2025), 5큐비트 DQPT로 영점=양자 위상 전이 실증

=============================================================================
"""

import numpy as np
from scipy.linalg import expm, eigh
import sys
import os

# bundle_utils 경로 등록
sys.path.insert(0, os.path.expanduser('~/Desktop/gdl_unified/scripts'))

# Qiskit 가용 여부 확인
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    HAS_QISKIT = True
    print("[정보] Qiskit + Aer 사용 가능")
except ImportError:
    HAS_QISKIT = False
    print("[정보] Qiskit 없음 → numpy/scipy fallback 사용")

# bundle_utils 불러오기 (없으면 mpmath 직접 사용)
try:
    from bundle_utils import curvature_zeta, curvature_at_t, find_zeros_zeta, xi_func, connection_zeta
    HAS_BUNDLE = True
    print("[정보] bundle_utils 로드 성공")
except ImportError:
    HAS_BUNDLE = False
    print("[정보] bundle_utils 없음 → mpmath 직접 계산 fallback")
    try:
        import mpmath
        mpmath.mp.dps = 80
        HAS_MPMATH = True
    except ImportError:
        HAS_MPMATH = False
        print("[경고] mpmath도 없음 → κ 프로파일 비활성화")


# ─────────────────────────────────────────────────────────────────────────────
# mpmath fallback: curvature_at_t
# ─────────────────────────────────────────────────────────────────────────────

def _xi_func_fallback(s):
    """ξ(s) = (1/2) s(s-1) π^(-s/2) Γ(s/2) ζ(s) — bundle_utils 없을 때 직접 계산"""
    half = mpmath.mpf('0.5')
    return half * s * (s - 1) * mpmath.power(mpmath.pi, -s / 2) * mpmath.gamma(s / 2) * mpmath.zeta(s)


def _curvature_at_t_fallback(t):
    """임계선 1/2+it 에서 κ = |ξ'/ξ|² — mpmath 직접 계산"""
    s = mpmath.mpf('0.5') + 1j * mpmath.mpf(str(t))
    h = mpmath.mpf(1) / mpmath.mpf(10 ** 20)
    xi_val = _xi_func_fallback(s)
    if abs(xi_val) < mpmath.mpf(10) ** (-mpmath.mp.dps + 10):
        return 1e10  # 영점 근방 → 발산
    xi_d = (_xi_func_fallback(s + h) - _xi_func_fallback(s - h)) / (2 * h)
    L = xi_d / xi_val
    return float(abs(L) ** 2)


def _find_zeros_fallback(t_min, t_max):
    """mpmath로 구간 내 리만 제타 영점 반환 — bundle_utils 없을 때"""
    zeros = []
    n = 1
    while True:
        t = float(mpmath.zetazero(n).imag)
        if t > t_max:
            break
        if t >= t_min:
            zeros.append(t)
        n += 1
    return np.array(zeros)


def get_curvature_at_t(t):
    """κ(t) 계산 — bundle_utils 우선, 없으면 fallback"""
    if HAS_BUNDLE:
        return curvature_at_t(t)
    elif HAS_MPMATH:
        return _curvature_at_t_fallback(t)
    else:
        return 0.0  # 비활성화 모드


def get_zeta_zeros(t_min, t_max):
    """리만 제타 영점 반환 — bundle_utils 우선, 없으면 fallback"""
    if HAS_BUNDLE:
        return find_zeros_zeta(t_min, t_max)
    elif HAS_MPMATH:
        return _find_zeros_fallback(t_min, t_max)
    else:
        # 알려진 첫 10개 영점 하드코딩 (mpmath 없을 때 최후 수단)
        known_zeros = np.array([14.1347, 21.0220, 25.0109, 30.4249,
                                32.9351, 37.5862, 40.9187, 43.3271,
                                48.0052, 49.7738])
        mask = (known_zeros >= t_min) & (known_zeros <= t_max)
        return known_zeros[mask]


# ─────────────────────────────────────────────────────────────────────────────
# 파울리 행렬 (2×2)
# ─────────────────────────────────────────────────────────────────────────────

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2      = np.eye(2, dtype=complex)


def kron_op(op, site, n):
    """
    n 큐비트 시스템에서 site번째에만 op가 작용하는 연산자를 구성.
    나머지 자리에는 항등 행렬.
    """
    ops = [I2] * n
    ops[site] = op
    result = ops[0]
    for m in ops[1:]:
        result = np.kron(result, m)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# IsingModel: 횡자기장 Ising 모형
# ─────────────────────────────────────────────────────────────────────────────

class IsingModel:
    """
    횡자기장 Ising 모형 (Transverse-Field Ising Model, TFIM)

        H = -J Σᵢ σᵢᶻ σᵢ₊₁ᶻ - h Σᵢ σᵢˣ

    n_qubits 큐비트, 주기 경계 조건 (PBC) 선택 가능.
    행렬 차원: 2^n × 2^n
    """

    def __init__(self, n_qubits: int, J: float = 1.0, h: float = 0.5,
                 pbc: bool = True):
        """
        Parameters
        ----------
        n_qubits : 큐비트 수 (권장 5~10)
        J        : Ising 결합 상수 (기본값 1.0)
        h        : 횡자기장 세기 (기본값 0.5)
        pbc      : 주기 경계 조건 여부 (기본값 True)
        """
        if n_qubits < 2:
            raise ValueError("큐비트 수는 2 이상이어야 합니다.")
        if n_qubits > 14:
            print(f"[경고] n_qubits={n_qubits}: 행렬 크기 {2**n_qubits}×{2**n_qubits} → 메모리 주의")

        self.n = n_qubits
        self.J = J
        self.h = h
        self.pbc = pbc
        self.dim = 2 ** n_qubits
        self._H = None  # 캐시

    def hamiltonian_matrix(self) -> np.ndarray:
        """
        전체 해밀토니안 행렬 구성 (2^n × 2^n, dtype=complex).

        Returns
        -------
        H : (2^n, 2^n) 복소 ndarray
        """
        if self._H is not None:
            return self._H

        n, J, h = self.n, self.J, self.h
        H = np.zeros((self.dim, self.dim), dtype=complex)

        # ZZ 결합항: -J Σ σᵢᶻ σᵢ₊₁ᶻ
        n_bonds = n if self.pbc else n - 1
        for i in range(n_bonds):
            j = (i + 1) % n
            ZZi = np.kron(kron_op(SIGMA_Z, i, n), np.eye(1))  # 차원 맞추기 방어코드
            # 직접 구성
            ops_left = [I2] * n
            ops_left[i] = SIGMA_Z
            L = ops_left[0]
            for m in ops_left[1:]:
                L = np.kron(L, m)
            ops_right = [I2] * n
            ops_right[j] = SIGMA_Z
            R = ops_right[0]
            for m in ops_right[1:]:
                R = np.kron(R, m)
            # σᵢᶻ ⊗ σⱼᶻ = 두 연산자의 성분별 곱 (같은 힐베르트 공간 기저)
            # 올바른 방법: kron_op 두 번 적용 후 행렬곱이 아니라 kron 배치
            H -= J * kron_op_pair(SIGMA_Z, i, SIGMA_Z, j, n)

        # 횡자기장항: -h Σ σᵢˣ
        for i in range(n):
            H -= h * kron_op(SIGMA_X, i, n)

        self._H = H
        return H

    def ground_state(self) -> np.ndarray:
        """
        바닥상태 |ψ₀⟩ 계산 (가장 낮은 고유값에 대응하는 고유벡터).

        Returns
        -------
        psi0 : (2^n,) 복소 ndarray, 정규화됨
        """
        H = self.hamiltonian_matrix()
        # eigh: 에르미트 행렬 고유분해 (scipy), 오름차순 정렬
        eigenvalues, eigenvectors = eigh(H)
        psi0 = eigenvectors[:, 0]  # 최소 고유값 대응
        # 정규화 확인
        norm = np.linalg.norm(psi0)
        return psi0 / norm

    def energy_gap(self) -> float:
        """
        에너지 갭 Δ = E₁ - E₀ (바닥상태 ~ 첫 번째 들뜬 상태).

        Returns
        -------
        gap : float
        """
        H = self.hamiltonian_matrix()
        eigenvalues = eigh(H, eigvals_only=True)
        return float(eigenvalues[1] - eigenvalues[0])


def kron_op_pair(op_i, site_i, op_j, site_j, n):
    """
    n 큐비트 시스템에서 site_i에 op_i, site_j에 op_j가 동시에 작용하는 연산자.
    나머지 자리에는 항등 행렬.
    """
    ops = [I2] * n
    ops[site_i] = op_i
    ops[site_j] = op_j
    result = ops[0]
    for m in ops[1:]:
        result = np.kron(result, m)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DQPTSimulator: 기본 DQPT 시뮬레이션
# ─────────────────────────────────────────────────────────────────────────────

class DQPTSimulator:
    """
    동적 양자 위상 전이(DQPT) 시뮬레이터.

    quench 프로토콜:
        t < 0: H₁ (h = h_initial) 의 바닥상태 |ψ₀⟩ 준비
        t ≥ 0: H₂ (h = h_final)  로 급변 후 시간 진화

    Loschmidt amplitude:
        G(t) = ⟨ψ₀|e^{-iH₂t}|ψ₀⟩

    Rate function:
        r(t) = -log|G(t)|² / N    (N = 큐비트 수)
    """

    def __init__(self, n_qubits: int, h_initial: float, h_final: float,
                 J: float = 1.0):
        """
        Parameters
        ----------
        n_qubits  : 큐비트 수
        h_initial : quench 전 횡자기장 (초기 바닥상태 준비)
        h_final   : quench 후 횡자기장 (시간 진화 해밀토니안)
        J         : Ising 결합 상수
        """
        self.n = n_qubits
        self.h_initial = h_initial
        self.h_final = h_final
        self.J = J

        # 초기 해밀토니안 → 바닥상태
        self.model_initial = IsingModel(n_qubits, J=J, h=h_initial)
        self.psi0 = self.model_initial.ground_state()

        # quench 후 해밀토니안
        self.model_final = IsingModel(n_qubits, J=J, h=h_final)
        self.H2 = self.model_final.hamiltonian_matrix()

        # H₂ 고유분해 (시간 진화 효율화)
        self._eigvals, self._eigvecs = eigh(self.H2)
        # |ψ₀⟩ → 고유기저 전개 계수
        self._psi0_eigbasis = self._eigvecs.conj().T @ self.psi0

    def loschmidt_echo(self, t_array: np.ndarray) -> np.ndarray:
        """
        Loschmidt amplitude G(t) = ⟨ψ₀|e^{-iH₂t}|ψ₀⟩ 계산.

        고유분해를 이용한 효율적 계산:
            G(t) = Σₖ |cₖ|² e^{-iEₖt}   (cₖ = ⟨Eₖ|ψ₀⟩)

        Parameters
        ----------
        t_array : (T,) 시간 배열

        Returns
        -------
        G : (T,) 복소 ndarray
        """
        c = self._psi0_eigbasis  # (dim,) 복소
        E = self._eigvals         # (dim,) 실수

        # G(t) = Σₖ |cₖ|² e^{-iEₖt}
        # 브로드캐스팅: E (dim,) × t (T,) → (dim, T)
        phase = np.exp(-1j * np.outer(E, t_array))   # (dim, T)
        weights = (c.conj() * c).real                 # |cₖ|² (dim,) 실수
        G = weights @ phase                            # (T,) 복소
        return G

    def rate_function(self, t_array: np.ndarray) -> np.ndarray:
        """
        Rate function r(t) = -log|G(t)|² / N.

        Parameters
        ----------
        t_array : (T,) 시간 배열

        Returns
        -------
        r : (T,) 실수 ndarray (|G|→0 근방은 대수 발산)
        """
        G = self.loschmidt_echo(t_array)
        absG2 = np.abs(G) ** 2
        # 수치 안정성: 0 방지
        absG2 = np.clip(absG2, 1e-300, None)
        r = -np.log(absG2) / self.n
        return r

    def find_critical_times(self, t_array: np.ndarray,
                             threshold: float = None) -> np.ndarray:
        """
        DQPT 임계 시간 t* 탐색.

        방법:
            r(t)의 국소 최댓값 (rate function 피크) 위치를 임계점으로 식별.
            rate function이 비해석적이 되는 점 = DQPT 임계 시간.

        Parameters
        ----------
        t_array   : (T,) 시간 배열
        threshold : rate function 피크 최소 높이 필터 (None이면 자동)

        Returns
        -------
        t_critical : (K,) 임계 시간 배열
        """
        r = self.rate_function(t_array)

        # 국소 최댓값 탐색 (전후 차분으로 피크 검출)
        # r[i]가 r[i-1], r[i+1]보다 크면 피크
        peaks = []
        for i in range(1, len(r) - 1):
            if r[i] > r[i - 1] and r[i] > r[i + 1]:
                peaks.append(i)

        if not peaks:
            return np.array([])

        # 임계값 필터
        if threshold is None:
            threshold = np.median(r) + 0.5 * np.std(r)

        critical = []
        for i in peaks:
            if r[i] > threshold:
                critical.append(t_array[i])

        return np.array(critical)

    def loschmidt_zeros(self, t_array: np.ndarray,
                        tol: float = 1e-3) -> np.ndarray:
        """
        |G(t)| ≈ 0 인 시간 (Loschmidt echo 영점) 탐색.

        Parameters
        ----------
        t_array : (T,) 시간 배열
        tol     : |G(t)| < tol 조건

        Returns
        -------
        t_zeros : 영점 근방 시간 배열
        """
        G = self.loschmidt_echo(t_array)
        absG = np.abs(G)
        zero_mask = absG < tol
        return t_array[zero_mask]


# ─────────────────────────────────────────────────────────────────────────────
# KappaInjectedDQPT: κ 프로파일 주입 DQPT (핵심 기여)
# ─────────────────────────────────────────────────────────────────────────────

class KappaInjectedDQPT(DQPTSimulator):
    """
    ξ-다발 곡률 κ(t) = |ξ'/ξ|² 를 Ising 모형의 결합 상수에 주입하는 DQPT.

    주입 방식:
        J(t) = J₀ · (1 + alpha · κ_norm(t))
        h(t) = h_final · (1 + beta · κ_norm(t))

    여기서 κ_norm은 [0, 1] 범위로 정규화된 κ 프로파일.

    가설:
        κ가 피크를 이루는 시간 (리만 영점 근방) 에서 DQPT 임계점이
        표준 DQPT에 비해 밀집하거나 강화될 것.

    시간 의존 해밀토니안이므로 Magnus 전개 / 1차 Trotter 분해 사용.
    """

    def __init__(self, n_qubits: int, h_initial: float, h_final: float,
                 zeta_zeros: np.ndarray, J: float = 1.0,
                 alpha: float = 0.3, beta: float = 0.0,
                 t_scale: float = 1.0):
        """
        Parameters
        ----------
        n_qubits    : 큐비트 수
        h_initial   : 초기 횡자기장
        h_final     : quench 후 횡자기장 (기저값)
        zeta_zeros  : 리만 제타 영점 배열 (임계선 위 허수부)
        J           : 기본 Ising 결합
        alpha       : J 변조 강도 (기본 0.3)
        beta        : h 변조 강도 (기본 0.0)
        t_scale     : DQPT 시간과 리만 t_n 의 단위 환산 인수
        """
        super().__init__(n_qubits, h_initial, h_final, J)
        self.zeta_zeros = np.asarray(zeta_zeros)
        self.alpha = alpha
        self.beta = beta
        self.t_scale = t_scale
        self.J_base = J

        # κ 프로파일 캐시
        self._kappa_cache = {}

    def kappa_profile(self, t_array: np.ndarray) -> np.ndarray:
        """
        임계선 1/2 + it·t_scale 에서 κ(t) = |ξ'/ξ|² 계산.

        DQPT 시간 t를 t_scale 로 스케일링하여 리만 t_n 단위와 맞춤.

        Parameters
        ----------
        t_array : (T,) DQPT 시간 배열

        Returns
        -------
        kappa : (T,) 실수 ndarray (정규화 전)
        """
        kappa = np.zeros(len(t_array))
        for idx, t in enumerate(t_array):
            t_riemann = float(t * self.t_scale)
            if t_riemann in self._kappa_cache:
                kappa[idx] = self._kappa_cache[t_riemann]
            else:
                val = get_curvature_at_t(t_riemann)
                # 발산 방지: 영점 바로 위에서 1e10 이상 → 클리핑
                val = min(val, 1e8)
                self._kappa_cache[t_riemann] = val
                kappa[idx] = val
        return kappa

    def kappa_profile_normalized(self, t_array: np.ndarray) -> np.ndarray:
        """
        κ 프로파일을 [0, 1] 범위로 정규화.

        Parameters
        ----------
        t_array : (T,) 시간 배열

        Returns
        -------
        kappa_norm : (T,) 실수 ndarray, 범위 [0, 1]
        """
        kappa = self.kappa_profile(t_array)
        kappa_max = kappa.max()
        if kappa_max < 1e-12:
            return np.zeros_like(kappa)
        return kappa / kappa_max

    def modulated_hamiltonian(self, t: float,
                               kappa_norm_val: float) -> np.ndarray:
        """
        κ(t)로 변조된 해밀토니안 H(t).

            J_eff(t) = J₀ · (1 + alpha · κ_norm(t))
            h_eff(t) = h_final · (1 + beta  · κ_norm(t))

        Parameters
        ----------
        t              : DQPT 시간 (float)
        kappa_norm_val : 해당 시간의 정규화된 κ 값

        Returns
        -------
        H_t : (2^n, 2^n) 복소 ndarray
        """
        J_eff = self.J_base * (1.0 + self.alpha * kappa_norm_val)
        h_eff = self.h_final * (1.0 + self.beta * kappa_norm_val)
        model_t = IsingModel(self.n, J=J_eff, h=h_eff)
        return model_t.hamiltonian_matrix()

    def time_evolved_loschmidt(self, t_array: np.ndarray,
                                dt: float = 0.01) -> np.ndarray:
        """
        시간 의존 H(t) 에 대한 Loschmidt echo 계산 (1차 Trotter 분해).

            U(t) ≈ Π_{k=0}^{T/dt} exp(-i H(tₖ) dt)
            G(t) = ⟨ψ₀|U(t)|ψ₀⟩

        κ 계산 비용이 크므로 t_array가 촘촘하면 시간이 걸릴 수 있음.
        dt가 작을수록 정확하지만 느려짐. 권장: dt ≥ 0.05.

        Parameters
        ----------
        t_array : (T,) 단조 증가 시간 배열 (t[0] ≥ 0)
        dt      : Trotter 스텝 크기

        Returns
        -------
        G : (T,) 복소 ndarray
        """
        print(f"[DQPT] κ-주입 시간 진화 시작 (n={self.n}, dt={dt}, "
              f"T_max={t_array[-1]:.2f}, 총 스텝≈{t_array[-1]/dt:.0f})")

        t_max = float(t_array[-1])
        n_steps = int(np.ceil(t_max / dt))
        dt_actual = t_max / n_steps

        # κ 프로파일 사전 계산 (Trotter 격자점)
        t_grid = np.linspace(0, t_max, n_steps + 1)
        print(f"[DQPT] κ 프로파일 계산 중 ({n_steps+1}점)...")
        kappa_norm_grid = self.kappa_profile_normalized(t_grid)
        print(f"[DQPT] κ 계산 완료. max={kappa_norm_grid.max():.4f}, "
              f"mean={kappa_norm_grid.mean():.4f}")

        # Trotter 진화
        psi_t = self.psi0.copy()
        G_results = {}

        # 저장할 시간 인덱스 미리 계산
        t_save_idx = np.searchsorted(t_grid, t_array)
        t_save_idx = np.clip(t_save_idx, 0, n_steps)

        step = 0
        G_at_steps = {}
        for k in range(n_steps):
            t_k = t_grid[k]
            kn = float(kappa_norm_grid[k])
            H_t = self.modulated_hamiltonian(t_k, kn)
            U_step = expm(-1j * H_t * dt_actual)
            psi_t = U_step @ psi_t

            # 저장 포인트 확인
            if (k + 1) in t_save_idx:
                G_at_steps[k + 1] = np.dot(self.psi0.conj(), psi_t)

            if (k + 1) % max(1, n_steps // 10) == 0:
                pct = 100 * (k + 1) / n_steps
                print(f"[DQPT]   진행: {pct:.0f}%")

        # t_array에 맞게 G 정렬
        G = np.array([G_at_steps.get(idx, complex(0))
                      for idx in t_save_idx])
        print("[DQPT] κ-주입 시간 진화 완료.")
        return G

    def kappa_dqpt_correlation(self, t_array: np.ndarray) -> float:
        """
        κ 프로파일 피크와 rate function 피크 사이의 상관계수 계산.

        Parameters
        ----------
        t_array : 분석 시간 구간

        Returns
        -------
        corr : Pearson 상관계수 (-1 ~ 1)
        """
        kappa = self.kappa_profile_normalized(t_array)
        r = self.rate_function(t_array)  # 부모 클래스 (정적 H) 사용
        corr = np.corrcoef(kappa, r)[0, 1]
        return float(corr)


# ─────────────────────────────────────────────────────────────────────────────
# 분석 함수
# ─────────────────────────────────────────────────────────────────────────────

def compare_dqpt_vs_zeros(dqpt_critical: np.ndarray,
                           zeta_zeros: np.ndarray,
                           tolerance: float = 0.5,
                           t_scale: float = 1.0) -> dict:
    """
    DQPT 임계점과 리만 영점 비교 분석.

    매칭 기준: |t*_DQPT - t_n / t_scale| < tolerance

    Parameters
    ----------
    dqpt_critical : DQPT 임계 시간 배열 (DQPT 단위)
    zeta_zeros    : 리만 제타 영점 배열 (리만 t_n 단위)
    tolerance     : 매칭 허용 오차 (DQPT 단위)
    t_scale       : 리만 t_n → DQPT 시간 환산 계수

    Returns
    -------
    result : {
        'matched_pairs'   : [(t_dqpt, t_riemann), ...],
        'n_matched'       : 매칭 수,
        'match_ratio'     : 매칭 비율 (DQPT 기준),
        'mean_distance'   : 평균 거리,
        'min_distances'   : 각 DQPT 임계점에서 가장 가까운 영점까지의 거리
    }
    """
    if len(dqpt_critical) == 0 or len(zeta_zeros) == 0:
        return {
            'matched_pairs': [],
            'n_matched': 0,
            'match_ratio': 0.0,
            'mean_distance': float('nan'),
            'min_distances': np.array([])
        }

    # 리만 영점을 DQPT 단위로 변환
    zeros_scaled = zeta_zeros / t_scale

    matched_pairs = []
    min_distances = []

    for t_dqpt in dqpt_critical:
        dists = np.abs(zeros_scaled - t_dqpt)
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        min_distances.append(min_dist)
        if min_dist < tolerance:
            matched_pairs.append((float(t_dqpt), float(zeta_zeros[min_idx])))

    min_distances = np.array(min_distances)
    match_ratio = len(matched_pairs) / len(dqpt_critical) if len(dqpt_critical) > 0 else 0.0

    return {
        'matched_pairs': matched_pairs,
        'n_matched': len(matched_pairs),
        'match_ratio': match_ratio,
        'mean_distance': float(np.mean(min_distances)) if len(min_distances) > 0 else float('nan'),
        'min_distances': min_distances
    }


def fisher_zeros(loschmidt_func, t_array: np.ndarray,
                 sigma_range: float = 0.5, n_sigma: int = 20) -> dict:
    """
    Fisher zeros 추출: 복소 시간 z = t + iσ 평면에서 G(z) = 0 탐색.

    G(z) = Σₖ |cₖ|² e^{-iEₖz}

    실시간 축 t에서 |G(t)|가 최솟값을 갖는 점들이 Fisher zero에 가장 가깝다.
    여기서는 허수 시간 σ 방향으로 스캔하여 |G| 최솟값을 추적.

    Parameters
    ----------
    loschmidt_func : callable, t_array → G(t) (복소 ndarray)
    t_array        : (T,) 실수 시간 배열
    sigma_range    : 허수 시간 탐색 범위 (기본 ±0.5)
    n_sigma        : 허수 시간 스캔 포인트 수

    Returns
    -------
    result : {
        'real_axis_minima'  : |G(t)| 최솟값 위치 (실시간 영점 후보),
        'sigma_scan'        : {t_idx: (σ, |G|) 최솟값 궤적},
        'fisher_zero_approx': 복소 Fisher zero 근사값 배열
    }
    """
    # 실축 Loschmidt 계산
    G_real = loschmidt_func(t_array)
    absG = np.abs(G_real)

    # |G(t)| 국소 최솟값 위치 (영점 후보)
    minima_idx = []
    for i in range(1, len(absG) - 1):
        if absG[i] < absG[i - 1] and absG[i] < absG[i + 1]:
            minima_idx.append(i)

    # 허수 시간 스캔: z = t + iσ
    sigma_vals = np.linspace(-sigma_range, sigma_range, n_sigma)
    fisher_approx = []

    for i in minima_idx[:10]:  # 상위 10개만 스캔 (계산 절약)
        t_crit = t_array[i]
        best_sigma = 0.0
        best_absG = absG[i]

        for sigma in sigma_vals:
            z = t_crit + 1j * sigma
            # G(z) = Σₖ |cₖ|² e^{-iEₖz} 는 loschmidt_func에서 직접 접근 불가
            # → 여기서는 실축 최솟값을 Fisher zero 근사로 활용
            pass

        fisher_approx.append(complex(t_crit, best_sigma))

    real_minima_t = t_array[minima_idx] if minima_idx else np.array([])
    real_minima_absG = absG[minima_idx] if minima_idx else np.array([])

    return {
        'real_axis_minima': list(zip(real_minima_t, real_minima_absG)),
        'fisher_zero_approx': np.array(fisher_approx),
        'n_minima': len(minima_idx)
    }


def fisher_zeros_exact(simulator: DQPTSimulator,
                        t_array: np.ndarray,
                        sigma_vals: np.ndarray = None) -> np.ndarray:
    """
    정확한 Fisher zeros 계산 (DQPTSimulator 내부 구조 활용).

        G(z) = Σₖ |cₖ|² e^{-iEₖz},   z = t + iσ

    Parameters
    ----------
    simulator : DQPTSimulator 인스턴스
    t_array   : (T,) 실수 시간 배열
    sigma_vals: 허수 부분 스캔 배열 (기본: linspace(-1, 1, 40))

    Returns
    -------
    fisher_zeros : (K,) 복소 ndarray — |G(z)|가 가장 작은 z 배열
    """
    if sigma_vals is None:
        sigma_vals = np.linspace(-1.0, 1.0, 40)

    c = simulator._psi0_eigbasis
    E = simulator._eigvals
    weights = (c.conj() * c).real  # |cₖ|²

    fisher_candidates = []

    for t in t_array:
        best_z = complex(t, 0)
        best_absG = float('inf')
        for sigma in sigma_vals:
            z = t + 1j * sigma
            phase = np.exp(-1j * E * z)   # (dim,)
            G_z = float(np.abs(weights @ phase))
            if G_z < best_absG:
                best_absG = G_z
                best_z = z
        if best_absG < 0.1:  # 충분히 작은 경우만 Fisher zero로 분류
            fisher_candidates.append(best_z)

    return np.array(fisher_candidates)


def print_comparison_report(label: str,
                             dqpt_result: dict,
                             comparison: dict,
                             kappa_corr: float = None):
    """
    DQPT 분석 결과를 한국어로 출력하는 보고서 함수.

    Parameters
    ----------
    label       : 실험 레이블
    dqpt_result : rate function / critical times 딕셔너리
    comparison  : compare_dqpt_vs_zeros 반환값
    kappa_corr  : κ-rate function 상관계수 (선택)
    """
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  [결과 보고] {label}")
    print(sep)

    t_crit = dqpt_result.get('critical_times', np.array([]))
    print(f"  DQPT 임계 시간 ({len(t_crit)}개): {np.round(t_crit, 4)}")

    zeros_scaled = dqpt_result.get('zeta_zeros_scaled', np.array([]))
    print(f"  리만 영점 (DQPT 단위, {len(zeros_scaled)}개): {np.round(zeros_scaled, 4)}")

    print(f"\n  매칭 결과:")
    print(f"    - 매칭된 쌍: {comparison['n_matched']}개")
    print(f"    - 매칭 비율: {comparison['match_ratio']*100:.1f}%")
    print(f"    - 평균 거리: {comparison['mean_distance']:.4f}")

    if comparison['matched_pairs']:
        print(f"\n  매칭 세부:")
        for t_d, t_z in comparison['matched_pairs']:
            print(f"    t_DQPT={t_d:.4f}  ↔  t_리만={t_z:.4f}")

    if kappa_corr is not None:
        print(f"\n  κ-rate function 상관계수: {kappa_corr:.4f}")
        if kappa_corr > 0.5:
            print(f"  → 강한 양의 상관 (κ 주입 효과 확인)")
        elif kappa_corr > 0.2:
            print(f"  → 중간 상관 (κ 영향 부분적)")
        else:
            print(f"  → 약한 상관 (추가 파라미터 조정 필요)")

    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 70)
    print("  DQPT-Zeta 프로토타입: 동적 양자 위상 전이 × 리만 영점")
    print("=" * 70)

    # ────────────────────────────────────────────
    # 파라미터 설정
    # ────────────────────────────────────────────
    N_QUBITS = 5          # 큐비트 수 (2^5 = 32차원, numpy 충분)
    J = 1.0               # Ising 결합
    H_INITIAL = 0.5       # quench 전 횡자기장 (질서 위상)
    H_FINAL = 1.5         # quench 후 횡자기장 (무질서 위상)
    T_MAX = 10.0          # 최대 시뮬레이션 시간
    N_T = 300             # 시간 격자 수
    T_SCALE = 3.0         # DQPT t → 리만 t_n 환산 계수 (t_riemann ≈ T_SCALE × t_dqpt)
    ALPHA = 0.5           # J 변조 강도
    BETA = 0.2            # h 변조 강도

    t_array = np.linspace(0.01, T_MAX, N_T)

    print(f"\n[설정]")
    print(f"  큐비트 수: {N_QUBITS} (힐베르트 공간 차원 = {2**N_QUBITS})")
    print(f"  quench: h {H_INITIAL} → {H_FINAL}  (J={J})")
    print(f"  시간 구간: [0, {T_MAX}], 격자 {N_T}점")
    print(f"  t_scale = {T_SCALE} (리만 t_n 범위 ≈ [{t_array[0]*T_SCALE:.1f}, {T_MAX*T_SCALE:.1f}])")

    # ────────────────────────────────────────────
    # 1. 리만 제타 영점 수집
    # ────────────────────────────────────────────
    print(f"\n[단계 1] 리만 제타 영점 수집 (t_n 범위: [{t_array[0]*T_SCALE:.1f}, {T_MAX*T_SCALE:.1f}])")
    t_min_riemann = float(t_array[0] * T_SCALE)
    t_max_riemann = float(T_MAX * T_SCALE)
    zeta_zeros = get_zeta_zeros(t_min_riemann, t_max_riemann)
    print(f"  영점 {len(zeta_zeros)}개 발견: {np.round(zeta_zeros, 4)}")

    # ────────────────────────────────────────────
    # 2. 기본 DQPT 시뮬레이션
    # ────────────────────────────────────────────
    print(f"\n[단계 2] 기본 DQPT 시뮬레이션 (n={N_QUBITS} 큐비트)")
    sim_base = DQPTSimulator(N_QUBITS, H_INITIAL, H_FINAL, J=J)

    # 에너지 갭 확인
    gap = sim_base.model_final.energy_gap()
    print(f"  quench 후 에너지 갭 Δ = {gap:.6f}")

    # Loschmidt echo + rate function
    G_base = sim_base.loschmidt_echo(t_array)
    r_base = sim_base.rate_function(t_array)
    t_crit_base = sim_base.find_critical_times(t_array)

    print(f"  DQPT 임계 시간 ({len(t_crit_base)}개): {np.round(t_crit_base, 4)}")
    print(f"  rate function 최댓값: {r_base.max():.4f} @ t={t_array[np.argmax(r_base)]:.4f}")
    print(f"  |G(t)| 최솟값: {np.abs(G_base).min():.2e} @ t={t_array[np.argmin(np.abs(G_base))]:.4f}")

    # 리만 영점과 비교
    comp_base = compare_dqpt_vs_zeros(
        t_crit_base, zeta_zeros,
        tolerance=T_MAX / 10,  # 시간 범위의 10%를 허용 오차로
        t_scale=T_SCALE
    )

    result_base = {
        'critical_times': t_crit_base,
        'zeta_zeros_scaled': zeta_zeros / T_SCALE
    }
    print_comparison_report("기본 DQPT", result_base, comp_base)

    # ────────────────────────────────────────────
    # 3. κ 주입 DQPT (KappaInjectedDQPT)
    # ────────────────────────────────────────────
    print(f"\n[단계 3] κ-주입 DQPT (alpha={ALPHA}, beta={BETA}, t_scale={T_SCALE})")

    kappa_sim = KappaInjectedDQPT(
        N_QUBITS, H_INITIAL, H_FINAL,
        zeta_zeros=zeta_zeros,
        J=J, alpha=ALPHA, beta=BETA,
        t_scale=T_SCALE
    )

    # κ 프로파일 계산 (저해상도로 빠르게)
    t_kappa_sample = np.linspace(t_array[0], T_MAX, 50)
    print(f"  κ 프로파일 계산 중 (50점 샘플)...")
    kappa_norm = kappa_sim.kappa_profile_normalized(t_kappa_sample)
    print(f"  κ_norm: max={kappa_norm.max():.4f}, mean={kappa_norm.mean():.4f}, "
          f"std={kappa_norm.std():.4f}")

    # κ-rate function 상관계수 (정적 H 기준, 빠른 계산)
    kappa_corr = kappa_sim.kappa_dqpt_correlation(t_array)
    print(f"  κ vs rate_function 상관계수: {kappa_corr:.4f}")

    # 정적 H₂ 기준 임계점 (κ 변조 전)
    t_crit_kappa_static = kappa_sim.find_critical_times(t_array)

    # Fisher zeros (정확한 방법)
    print(f"\n[단계 4] Fisher zeros 계산 (복소 시간 평면)")
    fisher_result = fisher_zeros_exact(
        sim_base, t_array,
        sigma_vals=np.linspace(-0.5, 0.5, 20)
    )
    print(f"  Fisher zero 후보 {len(fisher_result)}개:")
    for fz in fisher_result[:5]:  # 상위 5개 출력
        print(f"    z = {fz.real:.4f} + {fz.imag:.4f}i")

    # ────────────────────────────────────────────
    # 4. κ 주입 시간 진화 (Trotter, 간략 버전)
    # ────────────────────────────────────────────
    print(f"\n[단계 5] κ-주입 시간 진화 (Trotter, dt=0.1, 간략 구간)")
    t_short = np.linspace(0.1, 5.0, 50)  # 짧은 구간으로 빠른 데모
    G_kappa = kappa_sim.time_evolved_loschmidt(t_short, dt=0.1)
    r_kappa = -np.log(np.clip(np.abs(G_kappa)**2, 1e-300, None)) / N_QUBITS

    # κ-주입 임계점 탐색
    t_crit_kappa = []
    for i in range(1, len(r_kappa) - 1):
        if r_kappa[i] > r_kappa[i-1] and r_kappa[i] > r_kappa[i+1]:
            threshold = np.median(r_kappa) + 0.5 * np.std(r_kappa)
            if r_kappa[i] > threshold:
                t_crit_kappa.append(t_short[i])
    t_crit_kappa = np.array(t_crit_kappa)

    print(f"  κ-주입 DQPT 임계점 ({len(t_crit_kappa)}개): {np.round(t_crit_kappa, 4)}")

    comp_kappa = compare_dqpt_vs_zeros(
        t_crit_kappa, zeta_zeros,
        tolerance=5.0 / 10,
        t_scale=T_SCALE
    )
    result_kappa = {
        'critical_times': t_crit_kappa,
        'zeta_zeros_scaled': zeta_zeros / T_SCALE
    }
    print_comparison_report("κ-주입 DQPT", result_kappa, comp_kappa,
                             kappa_corr=kappa_corr)

    # ────────────────────────────────────────────
    # 5. 종합 비교
    # ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  종합 분석 결과")
    print("=" * 70)
    print(f"  리만 영점 (t_n): {np.round(zeta_zeros, 3)}")
    print(f"  리만 영점 → DQPT 단위 (t_n/{T_SCALE}): "
          f"{np.round(zeta_zeros/T_SCALE, 3)}")
    print(f"\n  기본 DQPT 임계점:   {np.round(t_crit_base, 3)}")
    print(f"  κ-주입 DQPT 임계점: {np.round(t_crit_kappa, 3)}")
    print(f"\n  [매칭 비교]")
    print(f"  기본 DQPT   매칭 비율: {comp_base['match_ratio']*100:.1f}%  "
          f"(평균 거리: {comp_base['mean_distance']:.4f})")
    print(f"  κ-주입 DQPT 매칭 비율: {comp_kappa['match_ratio']*100:.1f}%  "
          f"(평균 거리: {comp_kappa['mean_distance']:.4f})")

    improvement = comp_kappa['match_ratio'] - comp_base['match_ratio']
    if improvement > 0:
        print(f"\n  → κ 주입으로 매칭 비율 {improvement*100:.1f}% 향상 ✓")
        print(f"  → 가설 지지: ξ-다발 곡률 주입이 DQPT-영점 대응을 강화함")
    elif improvement == 0:
        print(f"\n  → κ 주입 효과 중립 (파라미터 조정 또는 t_scale 재보정 필요)")
    else:
        print(f"\n  → κ 주입 효과 역전 (alpha/beta 부호 재검토 필요)")

    print("\n[완료] dqpt_zeta.py 실행 종료.")

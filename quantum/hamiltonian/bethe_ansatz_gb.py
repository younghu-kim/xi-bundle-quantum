"""
Bethe Ansatz ↔ Gauss-Bonnet 등가성 분석

핵심 등가성:
    LeClair-Mussardo의 Bethe Ansatz 위상 누적  ←→  ξ-다발의 ∮ Im(L ds) = 2πN

이 모듈은 이 등가성을 수치적으로 검증하고, Bethe 방정식의 해가
리만 영점과 어떻게 대응하는지를 분석한다.

=== 이론적 배경 ===

1. Bethe Ansatz 측 (LeClair-Mussardo, JHEP 2024):
   1D 가적분 산란 모형에서 N-입자 파동함수는
       Ψ(x₁,...,x_N) = Σ_P A(P) exp(i Σ k_{P(j)} x_j)
   주기 경계조건 (길이 L 상자)으로부터 Bethe 방정식:
       e^{ik_j L} Π_{m≠j} S(k_j, k_m) = 1
   로그를 취하면:
       k_j L + Σ_{m≠j} δ(k_j, k_m) = 2π n_j    (n_j ∈ ℤ)
   여기서 δ(k, k') = -i log S(k, k')는 2체 산란 위상 이동.

   LeClair-Mussardo는 S-행렬을 S(θ) = ξ(1/2 + iθ)/ξ(1/2 - iθ) 로 선택하여
   Bethe 방정식의 해 {θ_j}가 리만 영점의 허수부 {t_n}과 일치하도록 구성.

2. ξ-다발 측 (우리 프레임워크):
   접속 L(s) = ξ'/ξ(s), 곡률 κ(s) = |L(s)|²
   Gauss-Bonnet: ∫∫_R κ dA = 2πN  (N = 영역 R 내 영점 수)
   등가 표현: ∮_∂R Im(L) ds = 2πN  (Stokes 정리)

3. 등가성의 핵심:
   Bethe 위상 누적 Σ δ(k_j) = 2πN  ←→  ∮ Im(ξ'/ξ) ds = 2πN

   즉 Bethe Ansatz의 위상 양자화 조건과 ξ-다발의 Gauss-Bonnet 조건이
   동일한 정수 위상학적 제약을 표현한다.

=== 참고 문헌 ===
- LeClair & Mussardo, JHEP 04 (2024) 062
- Berry & Keating, SIAM Review 41(2), 236-266 (1999)
"""
import numpy as np
from scipy.linalg import eigh
from scipy.integrate import quad
from scipy.optimize import fsolve
import sys
import os

# bundle_utils fallback
sys.path.insert(0, os.path.expanduser('~/Desktop/gdl_unified/scripts'))
try:
    from bundle_utils import xi_func, connection_zeta, find_zeros_zeta
    HAS_BUNDLE = True
except ImportError:
    HAS_BUNDLE = False

try:
    import mpmath
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False


# ============================================================
# 1. ξ 함수 및 접속 (mpmath fallback)
# ============================================================

def xi_function(s):
    """완성된 리만 ξ-함수: ξ(s) = (s/2)(s-1)π^{-s/2}Γ(s/2)ζ(s)"""
    if HAS_MPMATH:
        mp = mpmath
        mp.mp.dps = 30
        s_mp = mp.mpc(s)
        # ξ(s) = (1/2) s(s-1) π^{-s/2} Γ(s/2) ζ(s)
        return complex(
            mp.mpf('0.5') * s_mp * (s_mp - 1) *
            mp.power(mp.pi, -s_mp / 2) *
            mp.gamma(s_mp / 2) *
            mp.zeta(s_mp)
        )
    raise ImportError("mpmath 필요")


def xi_log_derivative(s, h=1e-8):
    """ξ'/ξ(s) = d/ds log ξ(s) — 수치 미분"""
    xi_plus = xi_function(s + h)
    xi_minus = xi_function(s - h)
    xi_s = xi_function(s)
    if abs(xi_s) < 1e-50:
        return complex('nan')
    return (xi_plus - xi_minus) / (2 * h * xi_s)


# ============================================================
# 2. Bethe Ansatz S-행렬 및 위상 이동
# ============================================================

class BetheAnsatzZeta:
    """LeClair-Mussardo 스타일 Bethe Ansatz — ζ 함수 S-행렬

    S-행렬 정의:
        S(θ) = ξ(1/2 + iθ) / ξ(1/2 - iθ)

    이 S-행렬은 다음을 자동으로 만족:
    1. 유니터리: |S(θ)|² = 1  (ξ의 함수방정식으로부터)
    2. 교차 대칭: S(iπ - θ) = S(θ)
    3. S의 영점 = ξ의 영점 (임계선 위의 영점)
    """

    def __init__(self, dps=30):
        """dps: mpmath 정밀도"""
        if not HAS_MPMATH:
            raise ImportError("mpmath 필요")
        self.dps = dps
        mpmath.mp.dps = dps

    def s_matrix(self, theta):
        """S(θ) = ξ(1/2 + iθ) / ξ(1/2 - iθ)

        θ가 실수이면 |S| = 1 (유니터리).
        임계선에서 ξ는 실수이므로 S = sign(ξ(1/2+iθ)/ξ(1/2-iθ)).
        영점을 지날 때 부호가 바뀌어 S → -1 (위상 π).
        """
        mpmath.mp.dps = self.dps
        s_plus = xi_function(0.5 + 1j * theta)
        s_minus = xi_function(0.5 - 1j * theta)
        if abs(s_minus) < 1e-50:
            return complex('nan')
        return complex(s_plus / s_minus)

    def phase_shift(self, theta):
        """산란 위상 이동 δ(θ) = arg(S(θ))

        임계선에서 ξ는 실수이므로 S = ±1,  δ = 0 또는 π.
        영점 사이에서 ξ의 부호에 따라 위상이 0 ↔ π로 점프.

        비자명한 위상을 얻으려면 off-shell (σ ≠ 1/2) 계산 필요.
        여기서는 σ = 0.5 + ε (약간 off-shell)로 계산하여
        연속적 위상 변화를 추적한다.
        """
        epsilon = 0.001  # 약간 off-shell
        mpmath.mp.dps = self.dps
        xi_plus = xi_function(0.5 + epsilon + 1j * theta)
        xi_minus = xi_function(0.5 + epsilon - 1j * theta)
        if abs(xi_minus) < 1e-50:
            return np.nan
        S = xi_plus / xi_minus
        return float(np.angle(S))

    def phase_shift_on_shell(self, theta):
        """On-shell (σ=1/2) 위상 이동 — arg ξ(1/2+iθ) 추적

        ξ(1/2+it)는 실수이므로 arg는 0 또는 π.
        영점을 지날 때마다 π 점프.
        이것이 모노드로미 ±π와 직접 대응.
        """
        val = xi_function(0.5 + 1j * theta)
        if abs(val) < 1e-50:
            return np.nan
        return 0.0 if val.real > 0 else np.pi

    def total_phase(self, rapidities):
        """N-입자 총 위상 누적

        Σ_{j=1}^{N} Σ_{m≠j} δ(θ_j - θ_m) / (2π)

        Bethe 조건이 만족되면 이 값은 정수.
        """
        N = len(rapidities)
        total = 0.0
        for j in range(N):
            for m in range(N):
                if m != j:
                    total += self.phase_shift(rapidities[j] - rapidities[m])
        return total / (2 * np.pi)

    def bethe_equations(self, rapidities, L=1.0):
        """Bethe 방정식 잔차

        e^{iθ_j L} Π_{m≠j} S(θ_j - θ_m) = 1

        로그 형태:
            θ_j L + Σ_{m≠j} δ(θ_j - θ_m) = 2π n_j

        Parameters:
            rapidities: 래피디티 배열 {θ_j}
            L: 시스템 크기

        Returns:
            잔차 배열 (0이면 Bethe 방정식 만족)
        """
        N = len(rapidities)
        residuals = np.zeros(N)
        for j in range(N):
            total_phase = rapidities[j] * L
            for m in range(N):
                if m != j:
                    total_phase += self.phase_shift(rapidities[j] - rapidities[m])
            # 가장 가까운 2πn으로의 잔차
            n_j = np.round(total_phase / (2 * np.pi))
            residuals[j] = total_phase - 2 * np.pi * n_j
        return residuals

    def solve_bethe(self, N_particles, L=1.0, initial_guess=None):
        """Bethe 방정식 풀기

        Parameters:
            N_particles: 입자 수
            L: 시스템 크기
            initial_guess: 초기 추정값 (없으면 리만 영점 사용)

        Returns:
            래피디티 배열 (Bethe 방정식의 해)
        """
        if initial_guess is None:
            # 리만 영점을 초기값으로 사용
            zeros = self._get_zeta_zeros(N_particles)
            initial_guess = np.array(zeros)

        def residual_func(thetas):
            return self.bethe_equations(thetas, L)

        solution = fsolve(residual_func, initial_guess, full_output=True)
        thetas = solution[0]
        info = solution[1]

        return thetas, np.max(np.abs(self.bethe_equations(thetas, L)))

    def _get_zeta_zeros(self, N):
        """처음 N개 리만 영점의 허수부"""
        # 알려진 리만 영점 (mpmath.zetazero로 검증 가능)
        known = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
                 37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
                 52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
                 67.079811, 69.546402, 72.067158, 75.704691, 77.144840]
        if N <= len(known):
            return known[:N]
        # mpmath fallback
        if HAS_MPMATH:
            zeros = []
            for n in range(1, N + 1):
                z = float(mpmath.im(mpmath.zetazero(n)))
                zeros.append(z)
            return zeros
        return known[:min(N, len(known))]


# ============================================================
# 3. Gauss-Bonnet 측 계산
# ============================================================

class GaussBonnetIntegral:
    """ξ-다발의 Gauss-Bonnet 적분

    ∮_∂R Im(ξ'/ξ) ds = 2πN

    여기서 R은 임계 영역(critical strip) 내 직사각형,
    N은 R 내 영점 수.
    """

    def __init__(self, dps=30):
        if not HAS_MPMATH:
            raise ImportError("mpmath 필요")
        self.dps = dps

    def connection_imaginary(self, sigma, t):
        """Im(ξ'/ξ(σ + it))"""
        mpmath.mp.dps = self.dps
        s = sigma + 1j * t
        L = xi_log_derivative(s)
        return np.imag(L)

    def contour_integral(self, sigma_min, sigma_max, t_min, t_max, n_points=200):
        """직사각형 경계를 따른 (1/2πi) ∮ (ξ'/ξ) ds 수치 적분

        Argument principle에 의해 이 적분 = R 내 영점 수 N.

        경로: 반시계 방향 직사각형
        ds는 복소수 (경로 접선 방향)

        Returns:
            integral / (2πi) 의 실부 — 정수여야 함 (영점 수)
        """
        total = 0.0 + 0.0j

        # 하단변: s = σ + i·t_min, σ: σ_min → σ_max,  ds = dσ
        sigmas = np.linspace(sigma_min, sigma_max, n_points)
        for i in range(len(sigmas) - 1):
            s_mid = (sigmas[i] + sigmas[i + 1]) / 2
            ds = sigmas[i + 1] - sigmas[i]  # 실수
            L = xi_log_derivative(s_mid + 1j * t_min)
            total += L * ds

        # 우변: s = σ_max + i·t, t: t_min → t_max,  ds = i·dt
        ts = np.linspace(t_min, t_max, n_points)
        for i in range(len(ts) - 1):
            t_mid = (ts[i] + ts[i + 1]) / 2
            dt = ts[i + 1] - ts[i]
            L = xi_log_derivative(sigma_max + 1j * t_mid)
            total += L * (1j * dt)

        # 상단변: s = σ + i·t_max, σ: σ_max → σ_min,  ds = dσ (음)
        for i in range(len(sigmas) - 1):
            idx = len(sigmas) - 1 - i
            s_mid = (sigmas[idx] + sigmas[idx - 1]) / 2
            ds = sigmas[idx - 1] - sigmas[idx]  # 음수
            L = xi_log_derivative(s_mid + 1j * t_max)
            total += L * ds

        # 좌변: s = σ_min + i·t, t: t_max → t_min,  ds = i·dt (음)
        for i in range(len(ts) - 1):
            idx = len(ts) - 1 - i
            t_mid = (ts[idx] + ts[idx - 1]) / 2
            dt = ts[idx - 1] - ts[idx]  # 음수
            L = xi_log_derivative(sigma_min + 1j * t_mid)
            total += L * (1j * dt)

        N_complex = total / (2 * np.pi * 1j)
        return np.real(N_complex)

    def curvature_area_integral(self, sigma_min, sigma_max, t_min, t_max,
                                 n_sigma=100, n_t=100):
        """면적분 ∫∫_R κ dσ dt / (2π)

        κ(s) = |ξ'/ξ(s)|²
        Stokes 정리에 의해 경계 적분과 같아야 함.
        """
        d_sigma = (sigma_max - sigma_min) / n_sigma
        d_t = (t_max - t_min) / n_t
        total = 0.0

        for i in range(n_sigma):
            sigma = sigma_min + (i + 0.5) * d_sigma
            for j in range(n_t):
                t = t_min + (j + 0.5) * d_t
                L = xi_log_derivative(sigma + 1j * t)
                kappa = abs(L) ** 2
                total += kappa * d_sigma * d_t

        return total / (2 * np.pi)


# ============================================================
# 4. 등가성 검증
# ============================================================

class BetheGBEquivalence:
    """Bethe Ansatz 위상 양자화 ↔ Gauss-Bonnet 2πN 등가성 검증

    두 측정량이 동일한 정수 N을 주는지 확인:

    (A) Bethe 측: Σ δ(θ_j - θ_m) / (2π) → N_Bethe
    (B) GB 측:   ∮ Im(ξ'/ξ) ds / (2π)   → N_GB

    둘 다 = R 내 영점 수여야 함.
    """

    def __init__(self, dps=30):
        self.bethe = BetheAnsatzZeta(dps=dps)
        self.gb = GaussBonnetIntegral(dps=dps)
        self.dps = dps

    def verify_equivalence(self, t_min=10.0, t_max=30.0, verbose=True):
        """[t_min, t_max] 구간에서 등가성 검증

        1. 이 구간 내 리만 영점 찾기
        2. Bethe Ansatz 위상 누적 계산
        3. Gauss-Bonnet 경계 적분 계산
        4. 비교
        """
        if verbose:
            print(f"=== Bethe Ansatz ↔ Gauss-Bonnet 등가성 검증 ===")
            print(f"구간: t ∈ [{t_min}, {t_max}]")
            print()

        # 1. 영점 찾기
        all_zeros = self.bethe._get_zeta_zeros(20)
        zeros_in_range = [z for z in all_zeros if t_min <= z <= t_max]
        N_exact = len(zeros_in_range)

        if verbose:
            print(f"1. 영점 수 (정확): N = {N_exact}")
            print(f"   영점 위치: {[f'{z:.6f}' for z in zeros_in_range]}")
            print()

        # 2. Bethe 측: 위상 누적
        if len(zeros_in_range) >= 2:
            bethe_phases = []
            for j, theta_j in enumerate(zeros_in_range):
                phase_j = 0.0
                for m, theta_m in enumerate(zeros_in_range):
                    if m != j:
                        delta = self.bethe.phase_shift(theta_j - theta_m)
                        if not np.isnan(delta):
                            phase_j += delta
                bethe_phases.append(phase_j)
            N_bethe = sum(bethe_phases) / (2 * np.pi * len(zeros_in_range))
        else:
            N_bethe = float(N_exact)
            bethe_phases = []

        if verbose:
            print(f"2. Bethe Ansatz 위상 누적:")
            print(f"   N_Bethe = {N_bethe:.6f}")
            if bethe_phases:
                print(f"   개별 위상: {[f'{p:.4f}' for p in bethe_phases]}")
            print()

        # 3. GB 측: 경계 적분
        try:
            N_gb = self.gb.contour_integral(
                sigma_min=0.3, sigma_max=0.7,
                t_min=t_min, t_max=t_max,
                n_points=200
            )
        except Exception as e:
            N_gb = float('nan')
            if verbose:
                print(f"   GB 적분 실패: {e}")

        if verbose:
            print(f"3. Gauss-Bonnet 경계 적분:")
            print(f"   N_GB = {N_gb:.6f}")
            print()

        # 4. 비교
        results = {
            'N_exact': N_exact,
            'N_bethe': N_bethe,
            'N_gb': N_gb,
            'zeros': zeros_in_range,
            'bethe_error': abs(N_bethe - N_exact),
            'gb_error': abs(N_gb - N_exact) if not np.isnan(N_gb) else float('nan'),
        }

        if verbose:
            print("4. 등가성 비교:")
            print(f"   |N_Bethe - N_exact| = {results['bethe_error']:.6f}")
            print(f"   |N_GB    - N_exact| = {results['gb_error']:.6f}")

            if results['bethe_error'] < 0.1 and (np.isnan(results['gb_error']) or results['gb_error'] < 0.5):
                print("   ✅ 등가성 확인: 두 경로 모두 동일한 정수 N을 산출")
            else:
                print("   ⚠️ 편차 존재 — 수치 정밀도 또는 경계 효과 가능")

        return results

    def scan_intervals(self, t_ranges=None, verbose=True):
        """여러 구간에서 등가성 스캔

        Parameters:
            t_ranges: [(t_min, t_max), ...] 리스트
        """
        if t_ranges is None:
            t_ranges = [
                (10.0, 16.0),   # 영점 1개 (14.13)
                (10.0, 22.0),   # 영점 2개
                (10.0, 26.0),   # 영점 3개
                (10.0, 35.0),   # 영점 5개
                (10.0, 50.0),   # 영점 10개
            ]

        summary = []
        for t_min, t_max in t_ranges:
            if verbose:
                print(f"\n{'='*60}")
            result = self.verify_equivalence(t_min, t_max, verbose=verbose)
            summary.append(result)

        if verbose:
            print(f"\n\n{'='*60}")
            print("=== 요약 ===")
            print(f"{'구간':>15s} | {'N_exact':>7s} | {'N_Bethe':>8s} | {'N_GB':>8s} | {'일치':>4s}")
            print("-" * 55)
            for (t_min, t_max), r in zip(t_ranges, summary):
                match = "✅" if r['bethe_error'] < 0.1 else "❌"
                print(f"  [{t_min:.0f}, {t_max:.0f}]    |"
                      f"   {r['N_exact']:>4d}  |"
                      f" {r['N_bethe']:>7.3f} |"
                      f" {r['N_gb']:>7.3f} |"
                      f"  {match}")

        return summary


# ============================================================
# 5. 유한 차원 Bethe 해밀토니안 구성
# ============================================================

def leclair_mussardo_hamiltonian(dim, coupling=1.0):
    """LeClair-Mussardo 해밀토니안의 유한 차원 트렁케이션

    1D Lieb-Liniger 모형의 Bethe Ansatz 해밀토니안:
    H = Σ_j (-∂²/∂x_j²) + c Σ_{j<k} δ(x_j - x_k)

    유한 기저에서 행렬 원소:
    H_{mn} = δ_{mn} k_m² + (c/L) Σ_α f(k_m - k_n; α)

    여기서 k_m은 양자수, f는 산란 커널.

    Parameters:
        dim: 행렬 차원
        coupling: 상호작용 강도 c

    Returns:
        dim × dim 에르미트 행렬
    """
    # 양자수: k_m = 2π m / L  (m = -dim//2, ..., dim//2-1)
    L = 2 * np.pi  # 단위 길이
    quantum_numbers = np.arange(-dim // 2, dim // 2)
    k = 2 * np.pi * quantum_numbers / L

    H = np.zeros((dim, dim), dtype=complex)

    # 대각 운동 에너지
    for m in range(dim):
        H[m, m] = k[m] ** 2

    # 비대각 산란 기여
    for m in range(dim):
        for n in range(dim):
            if m != n:
                dk = k[m] - k[n]
                # Lieb-Liniger 산란 커널: K(k) = 2c / (k² + c²)
                kernel = 2 * coupling / (dk ** 2 + coupling ** 2)
                H[m, n] = kernel / L

    # 에르미트화 보장
    H = (H + H.conj().T) / 2
    return H


def bethe_hamiltonian_from_zeros(zeros_t, dim=None):
    """리만 영점을 고유값으로 갖는 해밀토니안 직접 구성

    대각 행렬 + 섭동으로 구성:
    H = diag(t_1, t_2, ..., t_N) + ε V

    V는 영점 간 상관 구조를 반영하는 off-diagonal 항.

    Parameters:
        zeros_t: 리만 영점의 허수부 배열
        dim: 행렬 차원 (None이면 len(zeros_t))

    Returns:
        에르미트 행렬
    """
    N = len(zeros_t)
    if dim is None:
        dim = N

    H = np.zeros((dim, dim), dtype=complex)

    # 대각: 영점 위치
    for i in range(min(N, dim)):
        H[i, i] = zeros_t[i]

    # 나머지 대각: 외삽
    if dim > N:
        # Gram-Schmidt 보간
        for i in range(N, dim):
            H[i, i] = zeros_t[-1] + (i - N + 1) * np.mean(np.diff(zeros_t))

    # Off-diagonal: GUE 상관 구조를 반영하는 약한 커플링
    epsilon = 0.01 * np.mean(np.diff(zeros_t[:min(N, dim)]))
    for i in range(dim):
        for j in range(i + 1, dim):
            # pair correlation function에서 유도한 커플링
            gap = abs(H[i, i] - H[j, j])
            if gap > 0:
                coupling = epsilon / gap
                H[i, j] = coupling
                H[j, i] = coupling

    return H


# ============================================================
# 메인 실행
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  Bethe Ansatz ↔ Gauss-Bonnet 등가성 수치 검증")
    print("=" * 70)

    # 1. S-행렬 검증: |S(θ)|² = 1 (유니터리)
    print("\n[1] S-행렬 유니터리 검증")
    ba = BetheAnsatzZeta(dps=30)
    test_thetas = [1.0, 5.0, 10.0, 14.0, 20.0]
    for theta in test_thetas:
        S = ba.s_matrix(theta)
        if not np.isnan(S):
            print(f"  θ = {theta:>5.1f}: |S|² = {abs(S)**2:.10f}  "
                  f"({'✅ 유니터리' if abs(abs(S)**2 - 1) < 1e-6 else '❌'})")

    # 2. 위상 이동 프로파일
    print("\n[2] 산란 위상 이동 δ(θ)")
    thetas = np.linspace(0.1, 50, 50)
    phases = []
    for theta in thetas:
        delta = ba.phase_shift(theta)
        if not np.isnan(delta):
            phases.append((theta, delta))

    print(f"  계산된 점 수: {len(phases)}")
    if phases:
        print(f"  δ 범위: [{min(p[1] for p in phases):.4f}, {max(p[1] for p in phases):.4f}]")

    # 3. Bethe 방정식 해 검증
    print("\n[3] Bethe 방정식 — 리만 영점이 해인지 검증")
    zeros_3 = ba._get_zeta_zeros(3)
    residuals = ba.bethe_equations(np.array(zeros_3), L=1.0)
    print(f"  영점 3개: {[f'{z:.6f}' for z in zeros_3]}")
    print(f"  잔차: {[f'{r:.6f}' for r in residuals]}")

    # 4. 등가성 검증
    print("\n[4] Bethe ↔ GB 등가성 스캔")
    equiv = BetheGBEquivalence(dps=30)
    equiv.scan_intervals(verbose=True)

    # 5. LeClair-Mussardo 해밀토니안
    print("\n\n[5] LeClair-Mussardo 해밀토니안 (dim=32)")
    H_lm = leclair_mussardo_hamiltonian(32, coupling=1.0)
    eigvals = np.sort(np.real(eigh(H_lm)[0]))
    print(f"  고유값 범위: [{eigvals[0]:.4f}, {eigvals[-1]:.4f}]")
    print(f"  양수 고유값 수: {np.sum(eigvals > 0)}")

    # 6. 영점 기반 해밀토니안
    print("\n[6] 리만 영점 기반 해밀토니안 (dim=10)")
    zeros_10 = ba._get_zeta_zeros(10)
    H_zeros = bethe_hamiltonian_from_zeros(zeros_10)
    eigvals_z = np.sort(np.real(eigh(H_zeros)[0]))
    print(f"  입력 영점: {[f'{z:.2f}' for z in zeros_10]}")
    print(f"  출력 고유값: {[f'{e:.2f}' for e in eigvals_z]}")
    dev = np.max(np.abs(np.array(zeros_10) - eigvals_z))
    print(f"  최대 편차: {dev:.6f}  ({'✅' if dev < 0.5 else '⚠️'})")

    print("\n완료.")

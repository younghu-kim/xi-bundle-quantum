"""
Floquet 프로토콜을 이용한 리만 영점 측정 재현

USTC (2021)에서 포획 이온 1큐비트 + Floquet 드라이빙으로
처음 80개 리만 영점 위치를 측정한 실험의 수치 시뮬레이션.

=== 핵심 아이디어 ===

ξ(s) = ξ(1/2 + it) 를 시간에 의존하는 단일 큐비트 해밀토니안으로 인코딩:

    H(t) = Re[ξ(1/2+it)] σ_x + Im[ξ(1/2+it)] σ_y

|ψ₀⟩ = |0⟩에서 시작하여 H(t)로 시간 진화 → 측정.
Loschmidt echo |⟨0|ψ(t)⟩|²가 영점에서 0으로 떨어짐.

=== ξ-다발과의 연결 ===

1. H(t)의 구조는 ξ 함수의 실부/허부를 직접 인코딩
2. echo의 영점 = ξ의 영점 = 다발의 모노드로미 점
3. echo 주위 위상 변화 = 모노드로미 ±π와 대응
4. κ(t) = |ξ'/ξ|²가 클수록 echo의 하강이 급격 → 곡률 집중 확인 가능

=== 참고 ===
- USTC 실험: Luo et al., npj Quantum Information (2021)
- 우리 기여: κ 프로파일과 echo 급강하율의 상관 분석
"""
import numpy as np
from scipy.linalg import expm
import sys
import os

sys.path.insert(0, os.path.expanduser('~/Desktop/gdl_unified/scripts'))

try:
    import mpmath
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False

# Pauli 행렬
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


class XiFunctionEncoder:
    """ξ 함수를 단일 큐비트 해밀토니안으로 인코딩"""

    def __init__(self, dps=30):
        if not HAS_MPMATH:
            raise ImportError("mpmath 필요")
        self.dps = dps

    def xi_value(self, t):
        """ξ(1/2 + it) 계산"""
        mpmath.mp.dps = self.dps
        s = mpmath.mpc(0.5, t)
        val = (mpmath.mpf('0.5') * s * (s - 1) *
               mpmath.power(mpmath.pi, -s / 2) *
               mpmath.gamma(s / 2) *
               mpmath.zeta(s))
        return complex(val)

    def xi_derivative(self, t, h=1e-6):
        """ξ'(1/2 + it) 수치 미분"""
        val_plus = self.xi_value(t + h)
        val_minus = self.xi_value(t - h)
        return (val_plus - val_minus) / (2 * h)

    def hardy_z(self, t):
        """Hardy Z-함수: Z(t) = exp(iθ(t)) ξ(1/2+it) / |...| 정규화

        Z(t)는 실수이며 영점이 ξ와 동일.
        ζ(1/2+it)의 절대값에 비례하므로 O(1) 크기.
        """
        mpmath.mp.dps = self.dps
        # Z(t) = exp(i·theta(t)) * zeta(1/2+it)  where theta = Riemann-Siegel theta
        # siegeltheta가 없으면 직접 계산
        s = mpmath.mpc(0.5, t)
        theta_val = mpmath.im(mpmath.loggamma(s / 2)) - t / 2 * mpmath.log(mpmath.pi)
        Z = mpmath.exp(1j * theta_val) * mpmath.zeta(s)
        return float(mpmath.re(Z))

    def hamiltonian(self, t, scale=1.0):
        """H(t) = scale * Z(t) σ_x + scale * Z'(t) σ_y

        Z(t)는 Hardy Z-함수 (정규화, O(1) 크기).
        Z의 영점 = 리만 영점.
        """
        Z = self.hardy_z(t)
        Z_prime = (self.hardy_z(t + 1e-4) - self.hardy_z(t - 1e-4)) / 2e-4
        return scale * (Z * SIGMA_X + Z_prime * 0.01 * SIGMA_Y)

    def curvature(self, t):
        """κ(t) = |ξ'/ξ(1/2+it)|²"""
        xi_val = self.xi_value(t)
        xi_deriv = self.xi_derivative(t)
        if abs(xi_val) < 1e-50:
            return float('inf')
        L = xi_deriv / xi_val
        return abs(L) ** 2


class FloquetZeroDetector:
    """Floquet 프로토콜로 리만 영점 탐지

    시간 진화 연산자:
    U(t) = T exp(-i ∫₀ᵗ H(τ) dτ)

    Trotter 분해:
    U(t) ≈ Π_k exp(-i H(t_k) Δt)

    Loschmidt echo:
    L(t) = |⟨0|U(t)|0⟩|²
    """

    def __init__(self, encoder=None, dt=0.01, scale=0.01):
        """
        Parameters:
            encoder: XiFunctionEncoder 인스턴스
            dt: Trotter 시간 간격
            scale: 해밀토니안 스케일링 (큰 값이면 진동이 빨라짐)
        """
        self.encoder = encoder or XiFunctionEncoder()
        self.dt = dt
        self.scale = scale
        self.state_0 = np.array([1.0, 0.0], dtype=complex)  # |0⟩

    def time_evolve(self, t_start, t_end):
        """[t_start, t_end] 구간의 시간 진화 연산자 U"""
        n_steps = max(1, int((t_end - t_start) / self.dt))
        actual_dt = (t_end - t_start) / n_steps

        U = np.eye(2, dtype=complex)
        for k in range(n_steps):
            t_k = t_start + (k + 0.5) * actual_dt
            H_k = self.encoder.hamiltonian(t_k, scale=self.scale)
            U_k = expm(-1j * H_k * actual_dt)
            U = U_k @ U

        return U

    def loschmidt_echo(self, t_array, t_ref=0.0):
        """Loschmidt echo L(t) = |⟨0|U(t_ref→t)|0⟩|²

        점진적(incremental) 계산: U(t_{k+1}) = U_step · U(t_k)
        """
        # t_array가 정렬되어 있다고 가정
        sorted_idx = np.argsort(t_array)
        t_sorted = t_array[sorted_idx]
        echoes = np.zeros(len(t_array))

        # 기준 시간 이전은 1.0
        state = self.state_0.copy()
        current_t = t_ref

        for rank, orig_idx in enumerate(sorted_idx):
            t = t_sorted[rank]
            if t <= t_ref:
                echoes[orig_idx] = 1.0
                continue

            # current_t → t 까지 점진 진화
            n_steps = max(1, int((t - current_t) / self.dt))
            actual_dt = (t - current_t) / n_steps
            for k in range(n_steps):
                t_k = current_t + (k + 0.5) * actual_dt
                H_k = self.encoder.hamiltonian(t_k, scale=self.scale)
                U_k = expm(-1j * H_k * actual_dt)
                state = U_k @ state

            current_t = t
            echoes[orig_idx] = np.abs(np.vdot(self.state_0, state)) ** 2

        return echoes

    def rate_function(self, t_array, echoes=None, t_ref=0.0):
        """rate function r(t) = -log L(t)

        r(t)의 비해석적 점(cusp)이 DQPT 임계점.
        """
        if echoes is None:
            echoes = self.loschmidt_echo(t_array, t_ref)
        # 0 방지
        safe_echoes = np.maximum(echoes, 1e-30)
        return -np.log(safe_echoes)

    def find_echo_minima(self, t_array, echoes=None, threshold=0.1):
        """echo 극소점 탐색 — 영점 후보

        Parameters:
            t_array: 시간 배열
            echoes: 미리 계산된 echo (없으면 계산)
            threshold: 극소 판정 임계값

        Returns:
            (t_minima, echo_values) 튜플
        """
        if echoes is None:
            echoes = self.loschmidt_echo(t_array)

        minima_t = []
        minima_val = []

        for i in range(1, len(echoes) - 1):
            if echoes[i] < echoes[i - 1] and echoes[i] < echoes[i + 1]:
                if echoes[i] < threshold:
                    minima_t.append(t_array[i])
                    minima_val.append(echoes[i])

        return np.array(minima_t), np.array(minima_val)


class FloquetKappaCorrelation:
    """Floquet echo 극소점과 κ 프로파일의 상관 분석

    가설: κ(t)가 큰 영점 근방에서 echo의 하강이 더 급격하다.

    검증:
    1. 각 영점 t_n에서 echo의 하강률 계산: |dL/dt|_{t=t_n}
    2. 동일 t_n에서 κ(t_n) 계산
    3. 상관계수 r(|dL/dt|, κ) 계산
    """

    def __init__(self, detector, encoder=None):
        self.detector = detector
        self.encoder = encoder or detector.encoder

    def echo_descent_rate(self, t_zero, delta=0.5, n_points=50):
        """영점 t_zero 주위 echo 하강률

        [t_zero - delta, t_zero + delta] 구간에서
        echo의 최대 음의 기울기를 계산.
        """
        t_local = np.linspace(t_zero - delta, t_zero + delta, n_points)
        echoes = self.detector.loschmidt_echo(t_local, t_ref=t_zero - delta)

        # 수치 미분
        dt = t_local[1] - t_local[0]
        d_echo = np.diff(echoes) / dt

        return np.min(d_echo)  # 최대 하강 (음수)

    def curvature_at_zeros(self, zeros_t, offset=0.01):
        """각 영점에서의 곡률 (영점 바로 옆에서 측정)"""
        kappas = []
        for t_n in zeros_t:
            k = self.encoder.curvature(t_n + offset)
            kappas.append(k)
        return np.array(kappas)

    def correlation_analysis(self, zeros_t, verbose=True):
        """echo 하강률 vs κ 상관 분석

        Returns:
            dict with 'descent_rates', 'kappas', 'correlation'
        """
        if verbose:
            print("=== Floquet Echo ↔ κ 상관 분석 ===\n")

        descent_rates = []
        kappas = []

        for i, t_n in enumerate(zeros_t):
            rate = self.echo_descent_rate(t_n)
            kappa = self.encoder.curvature(t_n + 0.01)

            descent_rates.append(abs(rate))
            kappas.append(kappa)

            if verbose:
                print(f"  영점 #{i+1} (t={t_n:.4f}):"
                      f"  |dL/dt| = {abs(rate):.6f},"
                      f"  κ = {kappa:.2f}")

        descent_rates = np.array(descent_rates)
        kappas = np.array(kappas)

        # 유한 값만 사용
        mask = np.isfinite(descent_rates) & np.isfinite(kappas)
        if np.sum(mask) < 3:
            corr = float('nan')
        else:
            corr = np.corrcoef(descent_rates[mask], kappas[mask])[0, 1]

        if verbose:
            print(f"\n  상관계수 r(|dL/dt|, κ) = {corr:.4f}")
            if abs(corr) > 0.7:
                print("  ✅ 강한 상관: κ가 echo 하강을 지배")
            elif abs(corr) > 0.3:
                print("  ⚠️ 중간 상관: 부분적 영향")
            else:
                print("  ❌ 약한 상관: 독립적")

        return {
            'descent_rates': descent_rates,
            'kappas': kappas,
            'correlation': corr,
            'mask': mask,
        }


# ============================================================
# 메인 실행
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  Floquet 프로토콜 리만 영점 탐지 시뮬레이션")
    print("=" * 70)

    # 1. ξ 함수 인코더
    encoder = XiFunctionEncoder(dps=30)

    # 2. 처음 5개 영점 위치
    known_zeros = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062]
    print(f"\n[1] 리만 영점 (처음 5개): {[f'{z:.4f}' for z in known_zeros]}")

    # 3. Hardy Z 값 확인
    print("\n[2] 영점에서 Hardy Z 값:")
    for t in known_zeros:
        Z = encoder.hardy_z(t)
        print(f"  Z({t:.4f}) = {Z:.6e}  ({'≈0 ✅' if abs(Z) < 1e-6 else ''})")

    # 4. Floquet echo 계산
    print("\n[3] Loschmidt echo 계산 (t ∈ [12, 24])...")
    detector = FloquetZeroDetector(encoder, dt=0.1, scale=1.0)
    t_array = np.linspace(12, 24, 60)
    echoes = detector.loschmidt_echo(t_array, t_ref=10.0)

    # 5. 극소점 탐색
    minima_t, minima_val = detector.find_echo_minima(t_array, echoes, threshold=0.5)
    print(f"\n[4] Echo 극소점 (영점 후보): {len(minima_t)}개")
    for i, (t, v) in enumerate(zip(minima_t, minima_val)):
        # 가장 가까운 실제 영점 찾기
        dists = [abs(t - z) for z in known_zeros]
        nearest_idx = np.argmin(dists)
        print(f"  후보 #{i+1}: t = {t:.3f}, echo = {v:.4f},"
              f"  최근접 영점: {known_zeros[nearest_idx]:.4f}"
              f"  (거리: {dists[nearest_idx]:.3f})")

    # 6. Rate function
    rates = detector.rate_function(t_array, echoes)
    print(f"\n[5] Rate function r(t) 범위: [{np.min(rates):.4f}, {np.max(rates):.4f}]")

    # 7. κ 상관 분석
    print(f"\n[6] Echo 하강률 ↔ κ 상관 분석")
    corr_analyzer = FloquetKappaCorrelation(detector, encoder)
    corr_result = corr_analyzer.correlation_analysis(known_zeros[:5])

    # 8. 요약
    print(f"\n{'='*70}")
    print("  요약")
    print(f"{'='*70}")
    print(f"  탐지된 영점 후보: {len(minima_t)}개 / 실제 영점: {len(known_zeros)}개")
    print(f"  Echo-κ 상관: r = {corr_result['correlation']:.4f}")
    print(f"  ξ-다발 연결: 모노드로미 → echo 위상 점프 → DQPT 임계점")
    print()

import warnings
import numpy as np
import pytest
try:
    from p_kit.solver.spd_solver import SPDSolver,dense_jh,energy,ising_to_spd,mapping_quality,mapping_identity_error,target_corr
except ImportError:
    from spd_solver import SPDSolver,dense_jh,energy,ising_to_spd,mapping_quality,mapping_identity_error,target_corr

class ANDCircuit:
    J=np.array([[0.,-1.,2.],[-1.,0.,2.],[2.,2.,0.]])
    h=np.array([[1.],[1.],[-2.]])

class DummyFallback:
    def __init__(self,Nt=8,**kwargs): self.Nt=Nt; self.called=False
    def solve(self,circuit):
        self.called=True; _,h=dense_jh(circuit); n=len(h)
        return np.zeros((self.Nt,n)),np.ones((self.Nt,n))

def test_mapping_is_spd_and_exact():
    J,h=dense_jh(ANDCircuit()); m=ising_to_spd(J,h); q=mapping_quality(J,h,m)
    assert m.K.shape==(4,4)
    assert np.linalg.eigvalsh(m.K).min()>0
    assert q.score>.999999 and q.fidelity>.999999
    assert mapping_identity_error(J,h,m)<1e-10

def test_mapping_preserves_energy_ordering():
    J,h=dense_jh(ANDCircuit()); m=ising_to_spd(J,h); q=mapping_quality(J,h,m)
    assert q.rank>.999999
    assert q.nrmse<1e-12

def test_target_correlation_is_spd():
    J,h=dense_jh(ANDCircuit()); C=target_corr(ising_to_spd(J,h),len(h))
    assert np.allclose(C,C.T)
    assert np.allclose(np.diag(C),1.,atol=1e-7)
    assert np.linalg.eigvalsh(C).min()>0

def test_fast_solver_api_and_state():
    s=SPDSolver(Nt=300,seed=1,mode="fast",verbose=False); I,M=s.solve(ANDCircuit())
    assert I.shape==M.shape==(300,3)
    assert set(np.unique(M))<=set((-1.,1.))
    assert s.mapping_quality_.score>=s.min_mapping_score
    assert not s.used_fallback_
    assert 0<=s.acceptance_rate_<=1
    assert np.isfinite(s.energies_).all()

def test_analog_solver_api_and_spd_state():
    s=SPDSolver(Nt=80,seed=2,mode="analog",verbose=False); I,M=s.solve(ANDCircuit())
    assert I.shape==M.shape==(80,3)
    assert np.linalg.eigvalsh(s.manifold_state_).min()>0
    assert np.isfinite(s.final_distance_)

def test_fallback_instance_warns_and_runs():
    fb=DummyFallback(Nt=20); s=SPDSolver(Nt=20,max_condition=1.,min_mapping_score=.999,fall_back_solver=fb,verbose=False)
    with pytest.warns(RuntimeWarning,match="falling back to DummyFallback"):
        I,M=s.solve(ANDCircuit())
    assert s.used_fallback_ and fb.called
    assert I.shape==M.shape==(20,3)
    assert "mapping score" in s.fallback_reason_

def test_fallback_class_is_instantiated():
    s=SPDSolver(Nt=12,max_condition=1.,fall_back_solver=DummyFallback,verbose=False)
    with pytest.warns(RuntimeWarning,match="falling back"):
        I,M=s.solve(ANDCircuit())
    assert I.shape==M.shape==(12,3)
    assert isinstance(s.fallback_solver_,DummyFallback)

def test_low_score_without_fallback_raises():
    s=SPDSolver(Nt=10,max_condition=1.,min_mapping_score=.999,verbose=False)
    with pytest.raises(RuntimeError,match="mapping score"):
        s.solve(ANDCircuit())

@pytest.mark.parametrize("kwargs",[{"mode":"bad"},{"Nt":0},{"i0":0},{"margin":0},{"flip_prob":0},{"flip_prob":.5},{"min_mapping_score":1.1}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError): SPDSolver(**kwargs)

def test_and_truth_table_are_ground_states():
    J,h=dense_jh(ANDCircuit()); S=np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]],float)
    E=energy(S,J,h); ground={tuple(x.astype(int)) for x in S[np.isclose(E,E.min())]}
    expected={(-1,-1,-1),(-1,1,-1),(1,-1,-1),(1,1,1)}
    assert ground==expected

import numpy as np, sys
sys.path.insert(0, '.')
from src.data.domain_config import load_domain_config
from src.data.category_alert_generator import CategoryAlertGenerator
from src.models.profile_scorer import ProfileScorer

config = load_domain_config('soc_product_v50')
mu0 = config['mu'].copy()

gen = CategoryAlertGenerator(**config['generator_kwargs'], noise_rate=0.0, seed=42)
alerts = gen.generate(500)

scorer = ProfileScorer(mu0.copy(), config['actions'], tau=0.1, eta=0.05, eta_neg=0.05)
correct_before = sum(1 for a in alerts if scorer.score(a.factors, a.category_index).action_index == a.gt_action_index)

scorer2 = ProfileScorer(mu0.copy(), config['actions'], tau=0.1, eta=0.05, eta_neg=0.05)
for a in alerts:
    r = scorer2.score(a.factors, a.category_index)
    scorer2.update(a.factors, a.category_index, r.action_index, r.action_index == a.gt_action_index, gt_action_index=a.gt_action_index)
correct_after = sum(1 for a in alerts if scorer2.score(a.factors, a.category_index).action_index == a.gt_action_index)

drift = np.linalg.norm(scorer2.mu - mu0, axis=-1).mean()
print(f'Frozen:  {correct_before}/500 = {correct_before/5:.1f}%')
print(f'Learned: {correct_after}/500 = {correct_after/5:.1f}%')
print(f'Lift:    {(correct_after-correct_before)/5:+.1f}%')
print(f'Drift:   {drift:.4f}')
tag = 'PASS' if correct_after >= correct_before - 5 else 'FAIL'
print(f'Result:  {tag}')

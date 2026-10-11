"""Template demonstrating how a candidate machine learning model plugs into Stock Pulse.

Note: As per architectural rules, this file is an instructional template.
Candidate ML models should be registered and executed in subsequent phases.
"""
from sklearn.linear_model import Ridge
from src.models.contract import ModelSpec

def build_ridge_regression() -> Ridge:
    """Instantiate a sample regularized linear regression estimator."""
    return Ridge(alpha=1.0, random_state=42)

def get_template_spec() -> ModelSpec:
    """Return a ModelSpec instance configured for regression on cross-sectional ranks."""
    return ModelSpec(
        name="ridge_baseline_example",
        kind="regression",
        build_fn=build_ridge_regression,
        preprocessing="cs_rank",  # 'cs_rank' or 'raw'
        target_column=None        # Defaults to fwd_rankpct_{horizon}
    )

if __name__ == "__main__":
    print("Template model spec defined successfully:")
    spec = get_template_spec()
    print(f"Model: {spec.name}, Kind: {spec.kind}, Preprocessing: {spec.preprocessing}")

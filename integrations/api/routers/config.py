"""
Routers for configuration management.
Exposes endpoints for tuning pipeline hyperparameters.
"""

from fastapi import APIRouter, HTTPException

from meve import MeVeConfig
from ..schemas import ConfigUpdateRequest, MeVeConfigSchema

router = APIRouter(prefix="/config", tags=["configuration"])


@router.get("")
async def get_config(engine=None) -> MeVeConfigSchema:
    """
    Get current pipeline configuration.
    """
    if engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")

    try:
        config = engine.config if hasattr(engine, "config") else MeVeConfig()

        return MeVeConfigSchema(
            k_init=config.k_init,
            tau_relevance=config.tau_relevance,
            n_min=config.n_min,
            theta_redundancy=config.theta_redundancy,
            lambda_mmr=config.lambda_mmr,
            t_max=config.t_max,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("")
async def update_config(
    request: ConfigUpdateRequest,
    engine=None,
) -> dict:
    """
    Update pipeline configuration parameters.

    New configuration applies to subsequent queries.
    """
    if engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")

    try:
        # Create new config
        new_config = MeVeConfig(
            k_init=request.config.k_init,
            tau_relevance=request.config.tau_relevance,
            n_min=request.config.n_min,
            theta_redundancy=request.config.theta_redundancy,
            lambda_mmr=request.config.lambda_mmr,
            t_max=request.config.t_max,
        )

        # Update engine config
        if hasattr(engine, "config"):
            engine.config = new_config
        else:
            raise ValueError("Engine configuration update not supported")

        return {
            "status": "success",
            "message": "Configuration updated",
            "config": MeVeConfigSchema(
                k_init=new_config.k_init,
                tau_relevance=new_config.tau_relevance,
                n_min=new_config.n_min,
                theta_redundancy=new_config.theta_redundancy,
                lambda_mmr=new_config.lambda_mmr,
                t_max=new_config.t_max,
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/reset")
async def reset_config(engine=None) -> dict:
    """
    Reset configuration to defaults.
    """
    if engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")

    try:
        default_config = MeVeConfig()
        if hasattr(engine, "config"):
            engine.config = default_config

        return {
            "status": "success",
            "message": "Configuration reset to defaults",
            "config": MeVeConfigSchema(
                k_init=default_config.k_init,
                tau_relevance=default_config.tau_relevance,
                n_min=default_config.n_min,
                theta_redundancy=default_config.theta_redundancy,
                lambda_mmr=default_config.lambda_mmr,
                t_max=default_config.t_max,
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

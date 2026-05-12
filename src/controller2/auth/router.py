from fastapi import APIRouter
from fastapi.responses import RedirectResponse
from core.config import FRONTEND_URL, SUPABASE_URL
from core.supabase import supabase

router = APIRouter(prefix="/auth")


@router.get("/login")
def login():
    res = supabase.auth.sign_in_with_oauth({
        "provider": "google",
        "options": {
            "redirect_to": f"{SUPABASE_URL}/auth/v1/callback"
        }
    })
    return RedirectResponse(res.url)


@router.get("/callback")
def callback(code: str):
    res = supabase.auth.exchange_code_for_session({"auth_code": code}) # type: ignore

    access_token = res.session.access_token # type: ignore
    refresh_token = res.session.refresh_token # type: ignore

    return RedirectResponse(
        f"{FRONTEND_URL}?access_token={access_token}&refresh_token={refresh_token}"
    )
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    stream_with_context,
    url_for,
)
from graph.agent import FactCheckAgent
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker
from models import Base, VerificationMessage, VerificationSession
import json
import os
import sys
import uuid


app = Flask(__name__)
app.secret_key = os.urandom(24)

DATABASE_URL = "sqlite:///chat_history.db"
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def serialize_items(items):
    return [item.model_dump() if hasattr(item, "model_dump") else item for item in items]


app.template_folder = resource_path("templates")
app.static_folder = resource_path("static")


@app.route("/")
def index():
    return redirect(url_for("verify_page"))


@app.route("/new")
def new_verification():
    session.pop("chat_history", None)
    session.pop("model_provider", None)
    session.pop("model_name", None)
    session.pop("language", None)
    session.pop("session_id", None)
    return redirect(url_for("verify_page"))


@app.route("/verify", methods=["GET", "POST"])
def verify_page():
    if request.method == "GET":
        chat_history = session.get("chat_history", [])
        for msg in chat_history:
            if msg.get("type") == "ai":
                msg["content_html"] = msg.get("content", "")
        return render_template("verify.html", chat_history=chat_history)

    @stream_with_context
    def generate():
        db_session = SessionLocal()
        try:
            user_input = request.form.get("topic", "").strip()
            if not user_input:
                yield json.dumps(
                    {
                        "type": "error",
                        "message": "Please enter a news article URL, text, or claim.",
                    },
                    ensure_ascii=False,
                ) + "\n"
                return

            if "chat_history" not in session:
                model_provider = request.form.get("model_provider", "deepseek")
                model_name = request.form.get("model_name", "").strip() or None
                language = request.form.get("language", "zh")
                session["model_provider"] = model_provider
                session["model_name"] = model_name
                session["language"] = language
            else:
                model_provider = session.get("model_provider", "deepseek")
                model_name = session.get("model_name", None)
                language = session.get("language", "zh")

            session_id = session.get("session_id")
            verification_session = None

            if session_id:
                verification_session = (
                    db_session.query(VerificationSession).filter_by(id=session_id).first()
                )

            if verification_session is None:
                verification_session = VerificationSession(
                    id=session_id or str(uuid.uuid4()),
                    title=user_input[:200],
                )
                db_session.add(verification_session)
                db_session.commit()

            session_id = verification_session.id
            session["session_id"] = session_id

            chat_history = session.get("chat_history", [])
            chat_history.append({"type": "human", "content": user_input})
            session["chat_history"] = chat_history

            db_session.add(
                VerificationMessage(
                    session_id=session_id,
                    message_type="human",
                    content=user_input,
                )
            )
            db_session.commit()

            try:
                agent = FactCheckAgent(
                    model_provider=model_provider,
                    model_name=model_name,
                    language=language,
                )
                thread_id = session_id
                full_report = ""
                overall_verdict = None
                overall_confidence = None

                for chunk in agent.verify_stream(user_input, thread_id=thread_id):
                    for node_name, state_update in chunk.items():
                        _ = node_name

                        if "current_stage" in state_update:
                            stage_name = state_update["current_stage"]
                            stage_messages = {
                                "claim_extraction": "Extracting claims from input...",
                                "claim_decomposition": "Decomposing claims...",
                                "evidence_retrieval": "Searching for evidence...",
                                "source_credibility": "Evaluating source credibility...",
                                "evidence_aggregation": "Aggregating evidence...",
                                "multi_agent_debate": "Running multi-agent debate...",
                                "verdict_synthesis": "Generating verdict...",
                            }
                            yield json.dumps(
                                {
                                    "type": "stage",
                                    "stage": stage_name,
                                    "message": stage_messages.get(stage_name, stage_name),
                                },
                                ensure_ascii=False,
                            ) + "\n"

                        if "claims" in state_update and isinstance(
                            state_update["claims"], list
                        ):
                            yield json.dumps(
                                {
                                    "type": "claims",
                                    "data": serialize_items(state_update["claims"]),
                                },
                                ensure_ascii=False,
                            ) + "\n"

                        if "evidence" in state_update and isinstance(
                            state_update["evidence"], list
                        ):
                            yield json.dumps(
                                {
                                    "type": "evidence",
                                    "data": serialize_items(state_update["evidence"]),
                                },
                                ensure_ascii=False,
                            ) + "\n"

                        if "debate_arguments" in state_update and isinstance(
                            state_update["debate_arguments"], list
                        ):
                            yield json.dumps(
                                {
                                    "type": "debate",
                                    "data": serialize_items(
                                        state_update["debate_arguments"]
                                    ),
                                },
                                ensure_ascii=False,
                            ) + "\n"

                        if "verdicts" in state_update and isinstance(
                            state_update["verdicts"], list
                        ):
                            yield json.dumps(
                                {
                                    "type": "verdict",
                                    "data": serialize_items(state_update["verdicts"]),
                                },
                                ensure_ascii=False,
                            ) + "\n"

                        if "overall_verdict" in state_update:
                            overall_verdict = state_update["overall_verdict"]
                            overall_confidence = state_update.get(
                                "overall_confidence", overall_confidence or 0
                            )
                            yield json.dumps(
                                {
                                    "type": "overall_verdict",
                                    "verdict": overall_verdict,
                                    "confidence": overall_confidence,
                                },
                                ensure_ascii=False,
                            ) + "\n"

                        if "report_markdown" in state_update:
                            full_report = state_update["report_markdown"]
                            yield json.dumps(
                                {"type": "report", "markdown": full_report},
                                ensure_ascii=False,
                            ) + "\n"

                        if "errors" in state_update:
                            for err in state_update["errors"]:
                                yield json.dumps(
                                    {"type": "error", "message": str(err)},
                                    ensure_ascii=False,
                                ) + "\n"

                db_session.add(
                    VerificationMessage(
                        session_id=session_id,
                        message_type="ai",
                        content=full_report,
                    )
                )

                verification_session.overall_verdict = overall_verdict
                verification_session.overall_confidence = (
                    str(overall_confidence)
                    if overall_confidence is not None
                    else verification_session.overall_confidence
                )
                db_session.commit()

                chat_history.append({"type": "ai", "content": full_report})
                session["chat_history"] = chat_history

                yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"

            except Exception as e:
                db_session.rollback()
                yield json.dumps(
                    {"type": "error", "message": f"Error: {str(e)}"},
                    ensure_ascii=False,
                ) + "\n"
        finally:
            db_session.close()

    return Response(generate(), content_type="text/plain; charset=utf-8")


@app.route("/history")
def get_history():
    db_session = SessionLocal()
    try:
        sessions = db_session.query(VerificationSession).order_by(
            desc(VerificationSession.start_time)
        ).all()
        history_data = [
            {
                "id": item.id,
                "title": item.title
                or f"Verification at {item.start_time.strftime('%Y-%m-%d %H:%M')}",
                "start_time": item.start_time.isoformat(),
                "verdict": item.overall_verdict or "",
            }
            for item in sessions
        ]
        return jsonify(history_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db_session.close()


@app.route("/history/<session_id>")
def load_history(session_id):
    db_session = SessionLocal()
    try:
        session_data = db_session.query(VerificationSession).filter_by(id=session_id).first()
        if not session_data:
            return "Session not found", 404

        messages = (
            db_session.query(VerificationMessage)
            .filter_by(session_id=session_id)
            .order_by(VerificationMessage.timestamp)
            .all()
        )
        chat_history = [
            {"type": message.message_type, "content": message.content}
            for message in messages
        ]
        for msg in chat_history:
            if msg.get("type") == "ai":
                msg["content_html"] = msg.get("content", "")

        session["session_id"] = session_id
        session["chat_history"] = chat_history
        return render_template("verify.html", chat_history=chat_history)
    except Exception as e:
        return f"Error: {str(e)}", 500
    finally:
        db_session.close()


@app.route("/history_page")
def history_page():
    return render_template("history.html")


if __name__ == "__main__":
    app.run(debug=True, port=5001)

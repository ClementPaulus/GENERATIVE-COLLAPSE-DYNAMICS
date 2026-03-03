"""
Management dashboard pages: Notifications, Bookmarks, API Integration.
"""
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import json
from datetime import datetime

from umcp.dashboard._deps import pd, st
from umcp.dashboard._utils import (
    classify_regime,
    detect_anomalies,
    load_ledger,
)


def render_notifications_page() -> None:
    """Render the notification and alerts page."""
    if st is None:
        return

    st.title("🔔 Notifications & Alerts")
    st.caption("Configure alerts for regime changes and anomalies")

    # Initialize notification settings
    if "notifications" not in st.session_state:
        st.session_state.notifications = {
            "enabled": True,
            "regime_change": True,
            "anomaly_detected": True,
            "validation_failed": True,
            "threshold_omega_low": 0.1,
            "threshold_omega_high": 0.9,
            "alert_log": [],
        }

    # ========== Alert Configuration ==========
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚙️ Alert Settings")

        with st.container(border=True):
            st.session_state.notifications["enabled"] = st.toggle(
                "Enable Notifications",
                value=st.session_state.notifications["enabled"],
                help="Master switch for all alert types",
            )

            st.markdown("**Alert Types:**")
            st.session_state.notifications["regime_change"] = st.checkbox(
                "🌡️ Regime Changes",
                value=st.session_state.notifications["regime_change"],
                help="Alert when regime transitions (STABLE→WATCH, WATCH→COLLAPSE, etc.)",
            )
            st.session_state.notifications["anomaly_detected"] = st.checkbox(
                "⚠️ Anomaly Detection",
                value=st.session_state.notifications["anomaly_detected"],
                help="Alert when statistical anomalies are detected",
            )
            st.session_state.notifications["validation_failed"] = st.checkbox(
                "❌ Validation Failures",
                value=st.session_state.notifications["validation_failed"],
                help="Alert when validation returns NONCONFORMANT",
            )

            st.markdown("**Thresholds:**")
            st.session_state.notifications["threshold_omega_low"] = st.slider(
                "ω Low Threshold (COLLAPSE)",
                0.0,
                0.3,
                st.session_state.notifications["threshold_omega_low"],
                0.01,
                help="Alert when ω drops below this value",
            )
            st.session_state.notifications["threshold_omega_high"] = st.slider(
                "ω High Threshold (COLLAPSE)",
                0.7,
                1.0,
                st.session_state.notifications["threshold_omega_high"],
                0.01,
                help="Alert when ω exceeds this value",
            )

    with col2:
        st.subheader("🔍 Current State Check")

        if st.button("🔄 Check for Alerts Now", use_container_width=True):
            df = load_ledger()
            alerts = []

            if not df.empty:
                latest = df.iloc[-1]

                # Check omega thresholds
                if "omega" in df.columns:
                    omega = latest["omega"]
                    low_thresh = st.session_state.notifications["threshold_omega_low"]
                    high_thresh = st.session_state.notifications["threshold_omega_high"]

                    if omega < low_thresh:
                        alerts.append(
                            {
                                "type": "CRITICAL",
                                "message": f"ω below threshold: {omega:.4f} < {low_thresh}",
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                    elif omega > high_thresh:
                        alerts.append(
                            {
                                "type": "CRITICAL",
                                "message": f"ω above threshold: {omega:.4f} > {high_thresh}",
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

                # Check regime transitions
                if len(df) >= 2 and "omega" in df.columns:
                    prev = df.iloc[-2]
                    current_regime = classify_regime(latest.get("omega", 0.5), latest.get("seam_residual", 0))
                    prev_regime = classify_regime(prev.get("omega", 0.5), prev.get("seam_residual", 0))

                    if current_regime != prev_regime:
                        alerts.append(
                            {
                                "type": "WARNING",
                                "message": f"Regime changed: {prev_regime} → {current_regime}",
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

                # Check validation status
                if "run_status" in df.columns and latest["run_status"] == "NONCONFORMANT":
                    alerts.append(
                        {
                            "type": "ERROR",
                            "message": "Latest validation NONCONFORMANT",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                # Anomaly check
                if "omega" in df.columns and len(df) > 5:
                    anomalies = detect_anomalies(df["omega"])
                    if anomalies.iloc[-1]:
                        alerts.append(
                            {
                                "type": "WARNING",
                                "message": f"Statistical anomaly detected in ω: {latest['omega']:.4f}",
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

            # Display and log alerts
            if alerts:
                st.session_state.notifications["alert_log"].extend(alerts)
                for alert in alerts:
                    if alert["type"] == "CRITICAL":
                        st.error(f"🚨 **CRITICAL:** {alert['message']}")
                    elif alert["type"] == "ERROR":
                        st.error(f"❌ **ERROR:** {alert['message']}")
                    elif alert["type"] == "WARNING":
                        st.warning(f"⚠️ **WARNING:** {alert['message']}")
                    else:
                        st.info(f"ℹ️ **INFO:** {alert['message']}")
            else:
                st.success("✅ No alerts - system operating normally")

    # ========== Alert Log ==========
    st.divider()
    st.subheader("📜 Alert History")

    alert_log = st.session_state.notifications.get("alert_log", [])
    if alert_log:
        # Show last 20 alerts
        for alert in reversed(alert_log[-20:]):
            icon = (
                "🚨"
                if alert["type"] == "CRITICAL"
                else "⚠️"
                if alert["type"] == "WARNING"
                else "❌"
                if alert["type"] == "ERROR"
                else "ℹ️"
            )
            st.markdown(f"{icon} **{alert['type']}** — {alert['message']} @ {alert['timestamp'][:19]}")

        if st.button("🗑️ Clear Alert History"):
            st.session_state.notifications["alert_log"] = []
            st.rerun()
    else:
        st.info("No alerts recorded yet.")


def render_bookmarks_page() -> None:
    """Render the bookmarks page for saving interesting states."""
    if st is None or pd is None:
        return

    st.title("🔖 Bookmarks")
    st.caption("Save and revisit interesting states and configurations")

    # Initialize bookmarks
    if "bookmarks" not in st.session_state:
        st.session_state.bookmarks = []

    # ========== Add Bookmark ==========
    st.subheader("➕ Save Current State")

    with st.form("add_bookmark"):
        col1, col2 = st.columns(2)

        with col1:
            bookmark_name = st.text_input(
                "Bookmark Name", placeholder="e.g., 'Stable regime baseline'", help="Short descriptive name"
            )
            bookmark_type = st.selectbox(
                "Bookmark Type",
                ["Ledger Snapshot", "Configuration", "Audit Run", "Custom Note"],
                help="What kind of state to capture",
            )

        with col2:
            bookmark_tags = st.text_input("Tags (comma-separated)", placeholder="stable, baseline, v1.5")
            bookmark_notes = st.text_area("Notes", placeholder="Add any notes about this bookmark...")

        submitted = st.form_submit_button("🔖 Save Bookmark", use_container_width=True)

        if submitted and bookmark_name:
            # Capture current state
            df = load_ledger()
            snapshot = {}

            if bookmark_type == "Ledger Snapshot" and not df.empty:
                latest = df.iloc[-1].to_dict()
                # Convert numpy/pandas types to native Python
                snapshot = {
                    k: (float(v) if hasattr(v, "item") else str(v) if hasattr(v, "isoformat") else v)
                    for k, v in latest.items()
                    if pd.notna(v)
                }
            elif bookmark_type == "Audit Run" and "audit_log" in st.session_state and st.session_state.audit_log:
                snapshot = st.session_state.audit_log[-1]
            elif bookmark_type == "Configuration":
                snapshot = {
                    "auto_refresh": st.session_state.get("auto_refresh", False),
                    "refresh_interval": st.session_state.get("refresh_interval", 30),
                    "show_advanced": st.session_state.get("show_advanced", False),
                    "compact_mode": st.session_state.get("compact_mode", False),
                    "theme": st.session_state.get("theme", "Default"),
                }

            bookmark = {
                "id": len(st.session_state.bookmarks) + 1,
                "name": bookmark_name,
                "type": bookmark_type,
                "tags": [t.strip() for t in bookmark_tags.split(",") if t.strip()],
                "notes": bookmark_notes,
                "snapshot": snapshot,
                "created_at": datetime.now().isoformat(),
            }

            st.session_state.bookmarks.append(bookmark)
            st.success(f"✅ Bookmark '{bookmark_name}' saved!")
            st.rerun()

    st.divider()

    # ========== View Bookmarks ==========
    st.subheader("📚 Saved Bookmarks")

    if not st.session_state.bookmarks:
        st.info("No bookmarks saved yet. Create your first bookmark above!")
    else:
        # Filter by type
        types = list({b["type"] for b in st.session_state.bookmarks})
        type_filter = st.selectbox("Filter by Type", ["All", *types], help="Show only this bookmark type")

        filtered = st.session_state.bookmarks
        if type_filter != "All":
            filtered = [b for b in filtered if b["type"] == type_filter]

        # Display bookmarks
        for _i, bookmark in enumerate(reversed(filtered)):
            with st.expander(f"🔖 {bookmark['name']} ({bookmark['type']}) — {bookmark['created_at'][:10]}"):
                col1, col2 = st.columns([3, 1])

                with col1:
                    if bookmark["tags"]:
                        st.markdown("**Tags:** " + ", ".join([f"`{t}`" for t in bookmark["tags"]]))
                    if bookmark["notes"]:
                        st.markdown(f"**Notes:** {bookmark['notes']}")

                    st.markdown("**Snapshot:**")
                    st.json(bookmark["snapshot"])

                with col2:
                    if st.button("🗑️ Delete", key=f"del_bm_{bookmark['id']}"):
                        st.session_state.bookmarks = [
                            b for b in st.session_state.bookmarks if b["id"] != bookmark["id"]
                        ]
                        st.rerun()

                    # Export single bookmark
                    st.download_button(
                        label="📥 Export",
                        data=json.dumps(bookmark, indent=2, default=str),
                        file_name=f"bookmark_{bookmark['id']}.json",
                        mime="application/json",
                        key=f"export_bm_{bookmark['id']}",
                    )

        # Export all bookmarks
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Export All Bookmarks",
                data=json.dumps(st.session_state.bookmarks, indent=2, default=str),
                file_name=f"umcp_bookmarks_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True,
            )
        with col2:
            if st.button("🗑️ Clear All Bookmarks", use_container_width=True):
                st.session_state.bookmarks = []
                st.rerun()


# ============================================================================
# Medium-term Expansions - Time Series, Custom Formulas, Batch, API
# ============================================================================


def render_api_integration_page() -> None:
    """Render the API integration page for real-time sync."""
    if st is None:
        return

    st.title("🔌 API Integration")
    st.caption("Connect to the UMCP REST API for real-time data sync")

    # Initialize API settings
    if "api_settings" not in st.session_state:
        st.session_state.api_settings = {
            "url": "http://localhost:8000",
            "connected": False,
            "last_sync": None,
            "auto_sync": False,
        }

    # ========== Connection Settings ==========
    st.subheader("⚙️ Connection Settings")

    col1, col2 = st.columns([2, 1])

    with col1:
        api_url = st.text_input(
            "API URL",
            value=st.session_state.api_settings["url"],
            placeholder="http://localhost:8000",
            help="Base URL of the UMCP REST API",
        )
        st.session_state.api_settings["url"] = api_url

    with col2:
        if st.button("🔗 Test Connection", use_container_width=True):
            try:
                import urllib.error
                import urllib.request

                with urllib.request.urlopen(f"{api_url}/health", timeout=5) as response:
                    data = json.loads(response.read().decode())
                    st.session_state.api_settings["connected"] = True
                    st.success(f"✅ Connected! API Status: {data.get('status', 'OK')}")
            except urllib.error.URLError as e:
                st.session_state.api_settings["connected"] = False
                st.error(f"❌ Connection failed: {e.reason}")
            except Exception as e:
                st.session_state.api_settings["connected"] = False
                st.error(f"❌ Error: {e}")

    # Connection status
    status = "🟢 Connected" if st.session_state.api_settings["connected"] else "🔴 Disconnected"
    st.markdown(f"**Status:** {status}")

    st.divider()

    # ========== API Endpoints ==========
    st.subheader("📡 API Endpoints")

    if not st.session_state.api_settings["connected"]:
        st.warning("Connect to the API first to test endpoints.")
    else:
        tabs = st.tabs(["🏥 Health", "📒 Ledger", "📦 Casepacks", "✅ Validate"])

        with tabs[0]:
            st.markdown("### Health Check")
            if st.button("🔄 Fetch Health", key="api_health"):
                try:
                    import urllib.request

                    with urllib.request.urlopen(f"{api_url}/health", timeout=5) as response:
                        data = json.loads(response.read().decode())
                        st.json(data)
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        with tabs[1]:
            st.markdown("### Ledger Data")
            limit = st.slider("Limit", 5, 100, 20, key="api_ledger_limit", help="Max ledger rows to fetch")
            if st.button("🔄 Fetch Ledger", key="api_ledger"):
                try:
                    import urllib.request

                    with urllib.request.urlopen(f"{api_url}/ledger?limit={limit}", timeout=10) as response:
                        data = json.loads(response.read().decode())
                        if isinstance(data, list) and pd is not None:
                            df = pd.DataFrame(data)
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.json(data)
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        with tabs[2]:
            st.markdown("### Casepacks")
            if st.button("🔄 Fetch Casepacks", key="api_casepacks"):
                try:
                    import urllib.request

                    with urllib.request.urlopen(f"{api_url}/casepacks", timeout=10) as response:
                        data = json.loads(response.read().decode())
                        st.json(data)
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        with tabs[3]:
            st.markdown("### Validate via API")
            target = st.text_input("Target Path", value=".", key="api_validate_target")
            if st.button("🚀 Validate", key="api_validate"):
                try:
                    import urllib.request

                    req_data = json.dumps({"target": target}).encode()
                    req = urllib.request.Request(
                        f"{api_url}/validate",
                        data=req_data,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )

                    with urllib.request.urlopen(req, timeout=60) as response:
                        data = json.loads(response.read().decode())

                        if data.get("run_status") == "CONFORMANT":
                            st.success("✅ CONFORMANT")
                        else:
                            st.error("❌ NONCONFORMANT")

                        st.json(data)
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    st.divider()

    # ========== Auto-Sync ==========
    st.subheader("🔄 Auto-Sync")

    st.session_state.api_settings["auto_sync"] = st.toggle(
        "Enable Auto-Sync",
        value=st.session_state.api_settings["auto_sync"],
        help="Automatically sync data from API at regular intervals",
    )

    if st.session_state.api_settings["auto_sync"]:
        sync_interval = st.slider("Sync Interval (seconds)", 10, 120, 30)
        st.info(f"💡 Auto-sync will fetch data every {sync_interval} seconds when enabled.")

        if st.session_state.api_settings.get("last_sync"):
            st.caption(f"Last sync: {st.session_state.api_settings['last_sync']}")


# ============================================================================
# Precision Verification Page
# ============================================================================

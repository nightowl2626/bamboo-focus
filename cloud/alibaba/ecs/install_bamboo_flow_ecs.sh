#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/flowpilot}"
APP_PORT="${APP_PORT:-8000}"
PUBLIC_BASE_URL="${PUBLIC_BASE_URL:-}"
FLOWPILOT_TOKEN="${FLOWPILOT_TOKEN:-}"
QWEN_API_KEY="${QWEN_API_KEY:-${DASHSCOPE_API_KEY:-}}"
QWEN_API_BASE="${QWEN_API_BASE:-https://dashscope-intl.aliyuncs.com/compatible-mode/v1}"
QWEN_MODEL="${QWEN_MODEL:-qwen3.7-plus}"
NUDGE_MODE="${NUDGE_MODE:-auto}"
BASIC_AUTH_USER="${BASIC_AUTH_USER:-}"
BASIC_AUTH_PASSWORD="${BASIC_AUTH_PASSWORD:-}"

if [ "$(id -u)" -ne 0 ]; then
  echo "Run as root in Alibaba Cloud Workbench or with sudo." >&2
  exit 1
fi

if [ -z "$FLOWPILOT_TOKEN" ]; then
  echo "FLOWPILOT_TOKEN is required." >&2
  echo "Example: FLOWPILOT_TOKEN='long-random-token' PUBLIC_BASE_URL='http://1.2.3.4' bash cloud/alibaba/ecs/install_bamboo_focus_ecs.sh" >&2
  exit 1
fi

if [ -z "$BASIC_AUTH_USER" ] || [ -z "$BASIC_AUTH_PASSWORD" ]; then
  echo "BASIC_AUTH_USER and BASIC_AUTH_PASSWORD are required for any deploy reachable on a public IP." >&2
  echo "Without them, /app/ and every /api/* route (including the one that hands out FLOWPILOT_TOKEN) is open to the whole internet." >&2
  exit 1
fi

if [ -z "$PUBLIC_BASE_URL" ]; then
  PUBLIC_IP="$(curl -fsS --connect-timeout 2 http://100.100.100.200/latest/meta-data/eipv4 || true)"
  if [ -z "$PUBLIC_IP" ]; then
    PUBLIC_IP="$(curl -fsS --connect-timeout 2 http://100.100.100.200/latest/meta-data/public-ipv4 || true)"
  fi
  PUBLIC_BASE_URL="http://${PUBLIC_IP}"
fi

apt-get update
apt-get install -y python3 python3-venv python3-pip nginx curl

apt-get install -y apache2-utils
htpasswd -bc /etc/nginx/.bamboo_focus_htpasswd "$BASIC_AUTH_USER" "$BASIC_AUTH_PASSWORD"
if ! getent group www-data >/dev/null 2>&1; then
  echo "www-data group not found after installing nginx; refusing to leave the password file world-readable." >&2
  exit 1
fi
chown root:www-data /etc/nginx/.bamboo_focus_htpasswd
chmod 640 /etc/nginx/.bamboo_focus_htpasswd
AUTH_NGINX=$(printf '        auth_basic "Bamboo Focus";\n        auth_basic_user_file /etc/nginx/.bamboo_focus_htpasswd;')

cd "$APP_DIR"
python3 -m venv venv
./venv/bin/python -m pip install --upgrade pip

mkdir -p /var/log/flowpilot

cat >/etc/flowpilot.env <<ENVEOF
FLOWPILOT_PI_TOKEN=${FLOWPILOT_TOKEN}
FLOWPILOT_PUBLIC_BASE_URL=${PUBLIC_BASE_URL}
QWEN_API_KEY=${QWEN_API_KEY}
QWEN_API_BASE=${QWEN_API_BASE}
QWEN_MODEL=${QWEN_MODEL}
PYTHONUNBUFFERED=1
ENVEOF
chmod 600 /etc/flowpilot.env

id -u flowpilot >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin flowpilot
chown -R flowpilot:flowpilot "$APP_DIR" /var/log/flowpilot

cat >/etc/systemd/system/flowpilot.service <<SERVICEEOF
[Unit]
Description=Bamboo Focus ECS backend
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=flowpilot
Group=flowpilot
WorkingDirectory=${APP_DIR}
EnvironmentFile=/etc/flowpilot.env
ExecStart=${APP_DIR}/venv/bin/python ${APP_DIR}/app.py --host 127.0.0.1 --port ${APP_PORT} --nudge-mode ${NUDGE_MODE}
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
SERVICEEOF

cat >/etc/nginx/sites-available/flowpilot <<NGINXEOF
server {
    listen 80 default_server;
    server_name _;

    client_max_body_size 5m;

    location = /events {
        proxy_pass http://127.0.0.1:${APP_PORT};
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location = /pi/commands {
        proxy_pass http://127.0.0.1:${APP_PORT};
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location / {
${AUTH_NGINX}
        proxy_pass http://127.0.0.1:${APP_PORT};
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
NGINXEOF

rm -f /etc/nginx/sites-enabled/default
ln -sf /etc/nginx/sites-available/flowpilot /etc/nginx/sites-enabled/flowpilot
nginx -t
systemctl enable nginx
systemctl restart nginx
systemctl daemon-reload
systemctl enable flowpilot
systemctl restart flowpilot

echo ""
echo "Bamboo Focus is installed."
echo "Open: ${PUBLIC_BASE_URL}/app/"
echo ""
echo "Local webcam command:"
echo "python webcam_edge.py --laptop-api-base ${PUBLIC_BASE_URL} --token ${FLOWPILOT_TOKEN} --download-object-model --download-pose-model --debug-stream"
echo ""
echo "Logs:"
echo "journalctl -u flowpilot -f"

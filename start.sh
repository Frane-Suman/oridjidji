#!/usr/bin/env bash
# Pokretanje Oriđiđi demonstracije (Linux i macOS).
#
# Pokreće poslužitelj, čeka da se model učita i otvara sučelje u pregledniku.
# Zaustavljanje: Ctrl+C.
#
#   ./start.sh

set -euo pipefail
cd "$(dirname "$0")"

MODELI="$PWD/models"
ADRESA="http://127.0.0.1:8000"

if [ ! -f "$MODELI/basic/model/model.onnx" ]; then
    printf '\033[31mArtefakti modela nisu pronađeni u %s.\033[0m\n' "$MODELI"
    echo "Pokrenite najprije instalaciju:  bash install.sh"
    exit 1
fi

export ORIDJIDJI_CANDIDATE_ROOT="$MODELI"
export ORIDJIDJI_SKIP_BUNDLE_VERIFY=1

echo "Oriđiđi - demonstracija"
echo
printf '\033[90mNačin rada: osnovni. Učitavanje modela traje nekoliko desetaka sekundi,\033[0m\n'
printf '\033[90manaliza oko desetinke sekunde.\033[0m\n'
echo

uv run oridjidji serve --profile basic --candidate-root "$MODELI" --warmup &
POSLUZITELJ=$!
trap 'kill "$POSLUZITELJ" 2>/dev/null || true' EXIT INT TERM

printf 'Čekam da poslužitelj postane dostupan'
SPREMAN=0
for _ in $(seq 1 150); do
    if ! kill -0 "$POSLUZITELJ" 2>/dev/null; then
        echo
        printf '\033[31mPoslužitelj se zaustavio prije nego je postao dostupan.\033[0m\n'
        exit 1
    fi
    sleep 2
    if curl -fsS --max-time 3 "$ADRESA/api/v1/health" >/dev/null 2>&1; then
        SPREMAN=1
        break
    fi
    printf '.'
done
echo

if [ "$SPREMAN" -ne 1 ]; then
    printf '\033[31mPoslužitelj se nije javio unutar pet minuta.\033[0m\n'
    exit 1
fi

printf '\033[32mPoslužitelj je spreman.\033[0m Otvaram %s\n' "$ADRESA"
if command -v xdg-open >/dev/null 2>&1; then xdg-open "$ADRESA" >/dev/null 2>&1 || true
elif command -v open >/dev/null 2>&1; then open "$ADRESA" || true
else echo "Otvorite $ADRESA u pregledniku."; fi

echo
printf '\033[90mZa zaustavljanje pritisnite Ctrl+C.\033[0m\n'
wait "$POSLUZITELJ"

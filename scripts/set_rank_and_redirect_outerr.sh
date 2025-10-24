#!/usr/bin/bash
export RANK=$PALS_RANKID
ARGS=$@

case "$REDIRECT_OUTPUT_MODE" in
 "ALL_OE")
  $ARGS
 ;;
 "ALL_E")
  $ARGS 1>/dev/null
 ;;
 "Only_RANK0")
  if [[ $RANK -eq 0 ]]; then
    $ARGS
  else
    $ARGS 1>/dev/null 2>/dev/null
  fi
 ;;
esac
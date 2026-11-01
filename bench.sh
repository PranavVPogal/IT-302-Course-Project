#!/bin/bash
# bench.sh <label> <outfile.csv> <instance-list-file> <command...>
# Runs <command> <instance.vrp> -round R <extra> on every instance in the list and
# appends "label,instance,stats line" to the CSV. Golden*/CMT* use -round 0 (as in runRounds.sh).
label=$1; out=$2; list=$3; shift 3
cmd=$1; shift   # the instance file must come right after the binary
INP=${INP:-../parMDS/inputs}   # folder with the .vrp files (override with INP=...)
while read -r name; do
  [ -z "$name" ] && continue
  r=1; [[ $name == Golden* || $name == CMT* ]] && r=0
  line=$( ( ulimit -v 6500000; timeout ${TMO:-1800} $cmd $INP/$name.vrp -round $r "$@" 2>&1 >/dev/null ) | grep -v '^#' | tail -1)
  [ -z "$line" ] && line="FAILED"
  echo "$label,$name,$line" >> "$out"
done < "$list"

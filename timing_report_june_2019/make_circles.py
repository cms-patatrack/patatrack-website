#! /usr/bin/env python3
import sys

colours = {
  "AlCa":           "#ff9999",
  "B tagging":      "#663300",
  "E/Gamma":        "#ffee00",
  "ECAL":           "#4ddbff",
  "HCAL":           "#b5a642",
  "HLT":            "#808080",
  "I/O":            "#222222",
  "Jets/MET":       "#ee3300",
  "L1T":            "#cccccc",
  "Muons":          "#0040ff",
  "other":          "#ffffff",
  "Particle Flow":  "#ff66cc",
  "Pixels":         "#33ff33",
  "Taus":           "#800055",
  "Tracking":       "#009900",
  "Vertices":       "#006600",
}

# do not include modules and groups wiath a value less than or equal to the threshold
threshold = 0.001   # 1 us

def compute_sum(node):
  total = 0.
  for (label, value) in node.items():
    # skip special labels
    if label.startswith('@'):
      continue
    if type(value) is dict:
      total += compute_sum(value)
    else:
      total += value
  node['@total'] = total
  return total

def indent(level):
  sys.stdout.write('  ' * level)

def skip_node(label, node):
  # skip special nodes
  if label.startswith('@'):
    return True

  # skip nodes below threshold
  value = node['@total'] if type(node) is dict else node
  if value <= threshold:
    return True

  # otherwise, keep the node
  return False


def export_node(label, node, level = 0):
  value = node['@total'] if type(node) is dict else node
  indent(level)
  sys.stdout.write('{ "label": "%s", ' % label)
  if label in colours:
    sys.stdout.write('"color": "%s", ' % colours[label])
  if type(node) is dict:
    sys.stdout.write('"weight": %0.6g, "groups": [\n' % value)
    first = True
    for next_label, next_node in node.items():
      if skip_node(next_label, next_node):
        continue
      if not first:
        sys.stdout.write(',\n')
      first = False
      export_node(next_label, next_node, level + 1)
    sys.stdout.write('\n')
    indent(level)
    sys.stdout.write(']}')
  else:
    sys.stdout.write('"weight": %0.6g }' % value)

  return True


filename = sys.argv[1]
hierarchy = {}

for line in open(filename).readlines()[1:]:
  row = line.strip().split(',')
  if not row[0] in hierarchy:
    hierarchy[row[0]] = {}
  if not row[1] in hierarchy[row[0]]:
    hierarchy[row[0]][row[1]] = {}
  hierarchy[row[0]][row[1]][row[2]] = float(row[3])

title = 'HLT'
compute_sum(hierarchy)
export_node(title, hierarchy)


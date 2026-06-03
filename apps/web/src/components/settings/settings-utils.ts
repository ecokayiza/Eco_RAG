export function clampIndex(index: number, length: number) {
  if (length <= 0) {
    return 0;
  }
  return Math.min(Math.max(index, 0), length - 1);
}

export function resolveNextActiveName<T extends { name: string }>(
  items: T[],
  activeName: string | null,
  removedIndex: number
) {
  if (items.some((item) => item.name === activeName)) {
    return activeName ?? "";
  }
  return items[Math.max(0, removedIndex - 1)]?.name ?? items[0]?.name ?? "";
}

export function hasDuplicateName<T extends { name: string }>(items: T[], selectedIndex: number) {
  const item = items[selectedIndex];
  if (!item) {
    return false;
  }
  const name = item.name.trim().toLowerCase();
  return Boolean(name) && items.some((other, index) => index !== selectedIndex && other.name.trim().toLowerCase() === name);
}

export function getDuplicateNames<T extends { name: string }>(items: T[]) {
  const seen = new Set<string>();
  const duplicates = new Set<string>();

  for (const item of items) {
    const name = item.name.trim().toLowerCase();
    if (!name) {
      continue;
    }
    if (seen.has(name)) {
      duplicates.add(name);
      continue;
    }
    seen.add(name);
  }

  return [...duplicates];
}

export function optionalNumberValue(value: string) {
  if (!value.trim()) {
    return null;
  }
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function optionalPositiveIntegerValue(value: string) {
  if (!value.trim()) {
    return null;
  }
  const parsed = Number.parseInt(value, 10);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

export function optionalBooleanValue(value: string) {
  if (value === "true") {
    return true;
  }
  if (value === "false") {
    return false;
  }
  return null;
}

import { useEffect, useRef, useState } from 'react';
import { Button } from '@douyinfe/semi-ui';
import { getResourceFields, resourceLabel, setResourceValue, uploadResource } from './resource-files';

interface Props {
  value: unknown;
  onChange: (value: unknown) => void;
  directory?: boolean;
  label: string;
}

export function ResourceFilesInput({ value, onChange, directory = false, label }: Props) {
  const [busy, setBusy] = useState(false);
  return <ResourceFileChoice value={value} onChange={onChange} directory={directory} label={label}
    busy={busy} setBusy={setBusy} />;
}

function ResourceFileChoice({ value, onChange, directory = false, label, busy, setBusy }:
    Props & { busy: boolean; setBusy: (busy: boolean) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const current = useRef({ value, onChange });
  current.current = { value, onChange };
  const active = useRef<AbortController>();
  const [error, setError] = useState('');
  const [selected, setSelected] = useState<{ value: unknown; name: string }>();
  useEffect(() => () => {
    if (active.current) {
      active.current.abort();
      setBusy(false);
    }
  }, [setBusy]);

  async function choose(file?: File) {
    if (!file || busy) return;
    const original = value;
    const snapshot = JSON.stringify(value);
    const controller = new AbortController();
    active.current = controller;
    setBusy(true);
    setError('');
    try {
      const reference = await uploadResource(file, directory, controller.signal);
      if (!controller.signal.aborted && JSON.stringify(current.current.value) === snapshot) {
        const next = Array.isArray(original) ? [...original, reference] : reference;
        setSelected({ value: next, name: file.name });
        current.current.onChange(next);
      }
    } catch (failure) {
      if (!controller.signal.aborted) setError(failure instanceof Error ? failure.message : 'Model upload failed.');
    } finally {
      if (!controller.signal.aborted) setBusy(false);
      if (active.current === controller) active.current = undefined;
      if (input.current) input.current.value = '';
    }
  }

  return <div role="group" aria-label={label} style={{ minWidth: 210, maxWidth: 360, overflowWrap: 'anywhere' }}
    onClick={event => event.stopPropagation()} onPointerDown={event => event.stopPropagation()}>
    {Array.isArray(value) ? value.map((item, index) => <div key={index} style={{ marginBottom: 8 }}>
      <ResourceFileChoice label={`${label} ${index + 1}`} value={item} directory={directory} busy={busy} setBusy={setBusy}
        onChange={next => onChange(value.map((entry, position) => position === index ? next : entry))} />
      <Button size="small" aria-label={`Remove ${label} ${index + 1}`} disabled={busy}
        onClick={() => onChange(value.filter((_, position) => position !== index))}>Remove</Button>
    </div>) : <div style={{ marginBottom: 6 }}>{selected && selected.value === value ? selected.name : resourceLabel(value, directory)}</div>}
    <input ref={input} type="file" hidden aria-label={`Choose ${label}`} accept={directory ? '.zip' : undefined}
      disabled={busy} onChange={event => void choose(event.currentTarget.files?.[0])} />
    <Button size="small" loading={busy} disabled={busy} onClick={() => input.current?.click()}>
      {Array.isArray(value) ? 'Add file' : 'Choose file'}
    </Button>
    {!Array.isArray(value) && !!value && <Button size="small" disabled={busy} onClick={() => { setError(''); onChange(''); }}>Clear</Button>}
    {directory && <div style={{ marginTop: 4 }}>Model directory (.zip)</div>}
    {error && <div role="alert" style={{ color: 'var(--semi-color-danger)' }}>{error}</div>}
  </div>;
}

export function ResourceFieldsInput({ node, onChange }: { node: any; onChange: (node: any) => void }) {
  return <>{getResourceFields(node).map(field => <div key={JSON.stringify(field.path)} style={{ marginBottom: 8 }}>
    <div>{field.path.join(' / ')}</div>
    <ResourceFilesInput label={`${node.name_} ${field.path.join(' / ')}`} value={field.value} directory={field.directory}
      onChange={value => onChange(setResourceValue(node, field.path, value))} />
  </div>)}</>;
}

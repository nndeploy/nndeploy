import { createContext, FormEvent, ReactNode, useContext, useEffect, useState } from 'react';

const EditorSession = createContext(false);
const EditorAllowed = createContext(true);
export const useEditorSession = () => useContext(EditorSession);
export const useEditorAccessAllowed = () => useContext(EditorAllowed);

interface SessionStatus {
  required: boolean;
  authenticated: boolean;
  management_available: boolean;
}

export function EditorAccess({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<SessionStatus>();
  const [entered, setEntered] = useState(false);
  const [token, setToken] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  async function refresh() {
    const response = await fetch('/api/auth/session', { cache: 'no-store' });
    if (!response.ok) throw new Error('Cannot check editor access. Check the server connection and retry.');
    const value: SessionStatus = await response.json();
    if (typeof value.required !== 'boolean' || typeof value.authenticated !== 'boolean') {
      throw new Error('This editor requires an updated Server backend.');
    }
    setStatus(value);
    if (!value.required || value.authenticated) setEntered(true);
    setError('');
  }

  useEffect(() => {
    const check = () => { void refresh().catch(reason => setError(reason.message)); };
    check();
    const timer = window.setInterval(check, 30_000);
    window.addEventListener('focus', check);
    return () => { window.clearInterval(timer); window.removeEventListener('focus', check); };
  }, []);

  async function signIn(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    try {
      const response = await fetch('/api/auth/session', {
        method: 'POST', headers: { Authorization: `Bearer ${token}` },
      });
      if (!response.ok) {
        const value = await response.json();
        throw new Error(value.error?.message || 'Sign-in failed.');
      }
      setToken('');
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Sign-in failed.');
    } finally { setBusy(false); }
  }

  async function signOut() {
    setBusy(true);
    try {
      const response = await fetch('/api/auth/session', { method: 'DELETE' });
      if (!response.ok) throw new Error('Sign-out failed. Retry to end your session.');
      setToken('');
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Sign-out failed.');
    } finally { setBusy(false); }
  }

  const allowed = status && (!status.required || status.authenticated);
  return <EditorSession.Provider value={Boolean(allowed && status?.management_available)}><EditorAllowed.Provider value={Boolean(allowed)}>
    {entered && children}
    {allowed && status.authenticated && <button type="button" disabled={busy} onClick={signOut}
      style={{ position: 'fixed', top: 7, right: 82, zIndex: 2000 }}>Sign out</button>}
    {(!allowed || error) && <div style={{ position: 'fixed', inset: 0, zIndex: 10000,
      display: 'grid', placeItems: 'center', background: 'rgba(246,248,252,0.97)' }}>
      <form onSubmit={signIn} style={{ width: 380, maxWidth: '90vw', padding: 32,
        borderRadius: 12, background: 'white', boxShadow: '0 4px 24px #0001' }}>
        <h1 style={{ fontSize: 24 }}>nndeploy Server</h1>
        <p>{!status ? 'Connecting to the editor…' : 'Sign in to manage workflows, files and Server endpoints.'}</p>
        {status?.required && status.management_available && <>
          <label htmlFor="editor-management-token">Management token</label>
          <input id="editor-management-token" type="password" autoComplete="off" required value={token}
            onChange={event => setToken(event.target.value)} style={{ display: 'block', width: '100%', margin: '12px 0' }} />
          <button type="submit" disabled={busy || !token}>{busy ? 'Signing in…' : 'Sign in'}</button>
        </>}
        {status?.required && !status.management_available && <p>
          Editor access is disabled. Configure a management token when starting the Server.
        </p>}
        {error && <p role="alert">{error}</p>}
        {error && <button type="button" onClick={() => { void refresh().catch(reason => setError(reason.message)); }}>Retry</button>}
      </form>
    </div>}
  </EditorAllowed.Provider></EditorSession.Provider>;
}

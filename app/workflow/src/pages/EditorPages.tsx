import { useEffect, useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import Design from './Layout/Design/Design';

/** Keep canvas instances and unsaved edits alive while visiting Home. */
export default function EditorPages() {
  const location = useLocation();
  const visible = location.pathname === '/design';
  const [visited, setVisited] = useState(visible);
  useEffect(() => {
    if (visible) setVisited(true);
  }, [visible]);

  return <>
    <Outlet />
    {(visited || visible) && <div style={{ display: visible ? undefined : 'none' }} aria-hidden={!visible}>
      <Design visible={visible} />
    </div>}
  </>;
}

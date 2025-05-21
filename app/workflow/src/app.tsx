import { createRoot } from 'react-dom/client';

import { Editor } from './editor';

const app = createRoot(document.getElementById('root')!);

app.render(<Editor />);

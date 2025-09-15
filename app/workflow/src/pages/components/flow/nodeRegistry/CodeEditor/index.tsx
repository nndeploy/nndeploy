import { useRef } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
//import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { EnumIODataType } from '../ioType';
import { TextArea } from '@douyinfe/semi-ui';

import styles from './index.module.scss'



interface ICodEditorProps {
  value: string;
  ioDataType: EnumIODataType,
  direction: 'input' | 'output';
  onChange: (value: string) => void;
}

const CodeEditor: React.FC<ICodEditorProps> = (props) => {

  const { value, onChange } = props;
  // const handleInput = (e) => {
  //   const text = e.currentTarget.textContent;
  //   onChange(text);
  // };

  // Define language and code variables
  const language = 'javascript'; // or set based on ioDataType if needed
  return (
    <div style={{ position: 'relative' }}>
      <SyntaxHighlighter
        language={language}
        // style={atomDark} // Uncomment if you import atomDark
        customStyle={{
          margin: 0,
          padding: '1em',
          whiteSpace: 'pre-wrap',
          //whiteSpace: 'wrap',
          caretColor: 'red',

          wordBreak: 'break-all'
        }}
      >
        {value}
      </SyntaxHighlighter>

      <TextArea showClear value={value}

        className={styles['code-editor']}
        onChange={onChange}


        onClick={(event) => {
          event.stopPropagation();
        }} />
    </div>
  );
};

export default CodeEditor
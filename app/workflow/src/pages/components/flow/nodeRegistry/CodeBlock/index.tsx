import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
//import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';


interface ICodeBlockProps {
  code: string,
  language: string,
}

const CodeBlock: React.FC<ICodeBlockProps> = (props) => {
  const { code, language } = props;
  return (
    <SyntaxHighlighter
      language={language}
      //  style={atomDark}
      showLineNumbers
      wrapLines
    >
      {code}
    </SyntaxHighlighter>
  )
}



export default CodeBlock;
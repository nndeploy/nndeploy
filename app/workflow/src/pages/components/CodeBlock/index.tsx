import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import styles from './index.module.scss'
import classNames from 'classnames'
import 'github-markdown-css';

interface CodeBlockProps {
  code: string;
  language?: string;
}

export default function CodeBlock({ code, language }: CodeBlockProps) {
  return (
    <div className={classNames(styles['markdown-body'])}>
      <Markdown
        remarkPlugins={[remarkGfm]}

        components={{
          code({ node,
            //inline, 
            className, children, ...props }) {
            //const match = /language-(\w+)/.exec(className || '')
            return ( //  match && !inline 
              <SyntaxHighlighter
                style={vscDarkPlus}
                language={language}

                PreTag="div"
                wrapLines={true}
                customStyle={{
                  // background: 'transparent',
                  // margin: '12px 0',
                  // borderRadius: 8
                  background: '#f6f8fa',
                  color: '#24292e',
                  borderRadius: '6px',
                  padding: '16px',
                  fontSize: '85%',
                  lineHeight: 1.5
                }}
                codeTagProps={{
                  style: {
                    fontFamily: 'SFMono-Regular, Consolas, Liberation Mono, Menlo, monospace',
                    // 注释颜色调整
                    //color: '#6a737d!importent',
                    
                  }
                }}
              >
                {String(children).replace(/\n$/, '')}
              </SyntaxHighlighter>
            )
          }
        }}
      >
        {code}
      </Markdown>
    </div>
  )
}
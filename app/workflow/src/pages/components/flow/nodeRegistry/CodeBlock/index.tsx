import Markdown  from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import styles from './index.module.scss'

interface CodeBlockProps {
  code: string;
  language?: string;
}

export default function CodeBlock({ code, language }: CodeBlockProps) {
  return (
    <div className={styles['markdown-body']}>
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ node, 
            //inline, 
            className, children, ...props }) {
            //const match = /language-(\w+)/.exec(className || '')
            return  ( //  match && !inline 
              <SyntaxHighlighter
                style={vscDarkPlus}
                language={language}
                PreTag="div"
                wrapLines={true}
                customStyle={{
                  background: 'transparent',
                  margin: '12px 0',
                  borderRadius: 8
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
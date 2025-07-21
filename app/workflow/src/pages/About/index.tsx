
import styles from './index.module.scss'
export default function About() {
  return (
    <div className={styles['page-about']}>
      <div className={styles['parent']}>
            <div className={styles['child']}>child 1</div>
            <div className={styles['child']}>child 2</div>
      </div>
    </div>
  );
}
import { useEffect, useState, useRef } from 'react';

export function useGetRows(lineHeight: number = 20): [React.MutableRefObject<null>, number] {
  //const [size, setSize] = useState({ width: 0, height: 0 });
  const targetRef = useRef(null);

  const [rows, setRows] = useState(4)

  useEffect(() => {
    const observer = new ResizeObserver((entries) => {
      const {  height } = entries[0].contentRect;

      setRows(Math.floor(height / lineHeight))

    });

    if (targetRef.current) {
      observer.observe(targetRef.current);
    }

    return () => {
      if (targetRef.current) {
        observer.unobserve(targetRef.current);
      }
    };
  }, []);

  return [targetRef, rows];
}

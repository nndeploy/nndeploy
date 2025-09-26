import { useEffect, useState } from "react";
import { apiGetFileInfo } from "./api";
import { IServerResourceFileEntity } from "../../../../../Layout/Design/Resource/entity";

export function useGetFileInfo(filePath: string, direction: 'input' | 'output', runResult: string) {

  const [fileInfo, setFileInfo] = useState<IServerResourceFileEntity>({} as IServerResourceFileEntity);

  async function getFileInfo(filePath: string) {

    // const response = await fetch(filePath);
    // const text = await response.text();

    const response = await apiGetFileInfo(filePath)
    setFileInfo(response.result);
  }

  useEffect(() => {

    if (!filePath) {
      setFileInfo({} as IServerResourceFileEntity)
      return
    }

    if (direction == 'input') {

      getFileInfo(filePath)
    } else {
      if (runResult == 'success') {

         getFileInfo(filePath)
       
      } else {
        setFileInfo({} as IServerResourceFileEntity)
        return
      }
    }

  }, [filePath, direction, runResult])

  return fileInfo

}
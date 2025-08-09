import { useLocation } from "react-router-dom";

export function useQueryParams() {
  const { search } = useLocation();
  return new URLSearchParams(search);
}
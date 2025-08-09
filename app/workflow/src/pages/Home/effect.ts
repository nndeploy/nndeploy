import { useEffect, useState } from "react";
import { IBusinessNode } from "../Layout/Design/WorkFlow/entity";
import { apiGeTemplates, apiGetWorkflows } from "../../api";
import { IWorkFlowShortEntity } from "../../entity";

export function useGetTemplates() {
  const [templates, setTemplates] = useState<IWorkFlowShortEntity[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<any>(null);

  async function getTemplates() {
    setLoading(true);
    setError(null);
    try {
      var response = await apiGeTemplates();
      setTemplates(response.result);
    } catch (error) {
      setError(error);
    }
    setLoading(false);

  }
  useEffect(() => {
    getTemplates()
  }, [])

  return {
    templates,
    loading,
    error,
    getTemplates,
  }




}


export function useGetWorkflows() {
  const [workFlows, setWorkFlows] = useState<IWorkFlowShortEntity[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<any>(null);

  async function getTemplates() {
    setLoading(true);
    setError(null);
    try {
      var response = await apiGetWorkflows();
      setWorkFlows(response.result)
    } catch (error) {
      setError(error);
    }
    setLoading(false);

  }
  useEffect(() => {
    getTemplates()
  }, [])

  return {
    workFlows,
    loading,
    error,
    getTemplates,
  }




}

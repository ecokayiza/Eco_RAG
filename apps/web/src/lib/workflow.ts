import type { MetaResponse, WorkflowSnapshot } from "@/types/chat";

export function getWorkflowSteps(meta: MetaResponse | null) {
  if (!meta || meta.workflow_steps.length === 0) {
    throw new Error("Workflow metadata is missing workflow_steps.");
  }
  return meta.workflow_steps;
}

export function normalizeWorkflow(meta: MetaResponse | null, workflow: WorkflowSnapshot | null | undefined) {
  if (!workflow) {
    return null;
  }

  const nodeNames = getWorkflowSteps(meta);
  if (!workflow.node_statuses || workflow.node_statuses.length !== nodeNames.length) {
    throw new Error("Workflow snapshot is missing node_statuses.");
  }
  if (!workflow.logs) {
    throw new Error("Workflow snapshot is missing logs.");
  }
  if (
    workflow.status !== "completed" &&
    workflow.status !== "failed" &&
    workflow.status !== "stopped" &&
    workflow.active_node === undefined
  ) {
    throw new Error("Workflow snapshot is missing active_node.");
  }
  return workflow;
}

export function buildPendingWorkflow(meta: MetaResponse | null, query: string): WorkflowSnapshot {
  const nodeNames = getWorkflowSteps(meta);

  return {
    query,
    answer: "",
    status: "running",
    retrieve_round: 0,
    active_node: "plan",
    node_statuses: nodeNames.map((node) => ({
      node,
      status: node === "plan" ? "running" : "queued",
      detail: null,
    })),
    logs: [],
    errors: [],
  };
}

export function buildFailedWorkflow(workflow: WorkflowSnapshot, detail: string): WorkflowSnapshot {
  const errors = workflow.errors.includes(detail) ? workflow.errors : [...workflow.errors, detail];
  const hasLog = workflow.logs.some((entry) => entry.level === "error" && entry.message === detail);
  const logs = hasLog ? workflow.logs : [...workflow.logs, { level: "error", node: workflow.active_node, message: detail }];

  return {
    ...workflow,
    status: "failed",
    active_node: null,
    logs,
    errors,
    node_statuses: workflow.node_statuses.map((node) => {
      if (node.node === workflow.active_node) {
        return { ...node, status: "failed", detail };
      }
      return node;
    }),
  };
}

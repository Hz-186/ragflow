package entity

import "encoding/json"

// API4Conversation API for conversation model
type API4Conversation struct {
	ID           string          `gorm:"column:id;primaryKey;size:32" json:"id"`
	Name         *string         `gorm:"column:name;size:255" json:"name,omitempty"`
	DialogID     string          `gorm:"column:dialog_id;size:32;not null;index" json:"dialog_id"`
	UserID       string          `gorm:"column:user_id;size:255;not null;index" json:"user_id"`
	ExpUserID    *string         `gorm:"column:exp_user_id;size:255;index" json:"exp_user_id,omitempty"`
	Message      json.RawMessage `gorm:"column:message;type:longtext" json:"message,omitempty"`
	Reference    json.RawMessage `gorm:"column:reference;type:longtext" json:"reference,omitempty"`
	Tokens       int             `gorm:"column:tokens" json:"tokens"`
	Source       *string         `gorm:"column:source;size:16" json:"source,omitempty"`
	DSL          JSONMap         `gorm:"column:dsl;type:longtext" json:"dsl,omitempty"`
	Duration     float64         `gorm:"column:duration" json:"duration"`
	Round        int             `gorm:"column:round" json:"round"`
	ThumbUp      int             `gorm:"column:thumb_up" json:"thumb_up"`
	Errors       *string         `gorm:"column:errors;type:text" json:"errors,omitempty"`
	VersionTitle *string         `gorm:"column:version_title;size:255" json:"version_title,omitempty"`
	BaseModel
}

// TableName specify table name
func (API4Conversation) TableName() string {
	return "api_4_conversation"
}
